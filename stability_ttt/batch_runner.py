from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION

from .reference import ExpertActionReference
from .runner import (
    CONDITIONS,
    ProbeConfig,
    _camera_shift,
    _policy_batch,
)


def _noises(policy, seeds: list[int], step: int) -> torch.Tensor:
    tensors = []
    for seed in seeds:
        generator = torch.Generator(device=policy.config.device)
        generator.manual_seed(seed * 1_000_003 + step)
        tensors.append(
            torch.randn(
                (1, policy.config.chunk_size, policy.config.max_action_dim),
                generator=generator,
                device=policy.config.device,
            )
        )
    return torch.cat(tensors, dim=0)


def _env_actions(action, postprocessor, env_postprocessor) -> np.ndarray:
    transition = {ACTION: postprocessor(action)}
    result = env_postprocessor(transition)[ACTION]
    return result.detach().cpu().numpy().astype(np.float32)


class BatchedProbeRunner:
    """Seed-vectorized variant of ProbeRunner for camera/action/gripper shifts.

    It preserves one-step closed-loop inference. Only the model forward is batched;
    each simulator, residual, noise stream, termination flag, and log remains independent.
    Collision-aware object shifts intentionally use the sequential runner.
    """

    def __init__(self, cfg: ProbeConfig, batch_size: int):
        if cfg.object_dx_cm or cfg.object_dy_cm:
            raise ValueError("Object-pose shifts require the sequential collision-aware runner")
        self.cfg = cfg
        self.batch_size = batch_size
        self.policy = SmolVLAPolicy.from_pretrained(cfg.checkpoint).to(cfg.device).eval()
        policy_cfg = self.policy.config
        policy_cfg.device = cfg.device
        policy_cfg.n_action_steps = 1
        self.env_cfg = LiberoEnvConfig(task=cfg.suite, task_ids=list(cfg.task_ids), episode_length=cfg.max_steps)
        overrides = {"device_processor": {"device": cfg.device}}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=cfg.checkpoint,
            preprocessor_overrides=overrides,
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=self.env_cfg, policy_cfg=policy_cfg
        )
        self.envs = make_env(self.env_cfg, n_envs=batch_size, use_async_envs=False)
        self.references = {}

    def close(self):
        for env in self.envs[self.cfg.suite].values():
            env.close()

    def _reference(self, task_id, description):
        if task_id not in self.references:
            self.references[task_id] = ExpertActionReference(self.cfg.dataset_root, description)
        return self.references[task_id]

    def _reset(self, env, seeds):
        for subenv, seed in zip(env.envs, seeds, strict=True):
            subenv.init_state_id = seed % len(subenv._init_states)
        observation, _ = env.reset(seed=seeds)
        return observation

    @torch.inference_mode()
    def _actions_and_proxy(self, observation, description, seeds, step, need_proxy):
        shifted = _camera_shift(observation, self.cfg.camera_roll_deg)
        batch, states = _policy_batch(shifted, description, self.env_preprocessor, self.preprocessor)
        noise = _noises(self.policy, seeds, step)
        base = self.policy.predict_action_chunk(batch, noise=noise)[:, 0, :]
        proxy = torch.zeros_like(base)
        if need_proxy:
            augmented = _camera_shift(observation, self.cfg.camera_roll_deg + self.cfg.proxy_roll_deg)
            aug_batch, _ = _policy_batch(augmented, description, self.env_preprocessor, self.preprocessor)
            aug = self.policy.predict_action_chunk(aug_batch, noise=noise)[:, 0, :]
            proxy = aug - base
        return base, proxy, states

    def _calibration_pass(self, env, seeds, description):
        observation = self._reset(env, seeds)
        active = np.ones(len(seeds), dtype=bool)
        proxy_sum = torch.zeros((len(seeds), 7), device=self.cfg.device)
        proxy_count = torch.zeros((len(seeds), 1), device=self.cfg.device)
        max_steps = env.call("_max_episode_steps")[0]
        for step in range(max_steps):
            need_proxy = step % self.cfg.update_interval == 0
            base, proxy, _ = self._actions_and_proxy(observation, description, seeds, step, need_proxy)
            mask = torch.as_tensor(active, device=self.cfg.device).unsqueeze(1)
            if need_proxy:
                proxy_sum += proxy * mask
                proxy_count += mask
            actions = _env_actions(base, self.postprocessor, self.env_postprocessor)
            observation, _, terminated, truncated, _ = env.step(actions)
            active &= ~(terminated | truncated)
            if not active.any():
                break
        residual = self.cfg.update_eta * proxy_sum / proxy_count.clamp_min(1)
        return residual.clamp(-self.cfg.residual_clip, self.cfg.residual_clip)

    def run_batch(self, task_id: int, seeds: list[int], condition: str):
        env = self.envs[self.cfg.suite][task_id]
        description = env.call("task_description")[0]
        reference = self._reference(task_id, description)
        fixed = (
            self._calibration_pass(env, seeds, description)
            if condition == "buffer_offline"
            else torch.zeros((len(seeds), 7), device=self.cfg.device)
        )
        observation = self._reset(env, seeds)
        residual = fixed.clone()
        active = np.ones(len(seeds), dtype=bool)
        success = np.zeros(len(seeds), dtype=bool)
        logs = [[] for _ in seeds]
        delays = [deque(maxlen=max(1, self.cfg.gripper_delay_steps + 1)) for _ in seeds]
        max_steps = env.call("_max_episode_steps")[0]
        for step in range(max_steps):
            if condition == "online_reset" and step and step % self.cfg.reset_horizon == 0:
                residual.zero_()
            update = condition in ("online_persistent", "online_reset") and step % self.cfg.update_interval == 0
            base_norm, proxy, states = self._actions_and_proxy(observation, description, seeds, step, update)
            before = residual.clone()
            mask = torch.as_tensor(active, device=self.cfg.device).unsqueeze(1)
            if update:
                updated = residual * self.cfg.residual_decay + self.cfg.update_eta * proxy
                residual = torch.where(mask, updated, residual).clamp(-self.cfg.residual_clip, self.cfg.residual_clip)
            adapted_norm = base_norm if condition == "frozen" else base_norm + residual
            base_actions = _env_actions(base_norm, self.postprocessor, self.env_postprocessor)
            actions = _env_actions(adapted_norm, self.postprocessor, self.env_postprocessor)
            if self.cfg.action_bias:
                actions[:, :3] = np.clip(actions[:, :3] + self.cfg.action_bias, -1, 1)
            for i, delay in enumerate(delays):
                delay.append(float(actions[i, 6]))
                if self.cfg.gripper_delay_steps and len(delay) > self.cfg.gripper_delay_steps:
                    actions[i, 6] = delay[0]
            observation, reward, terminated, truncated, info = env.step(actions)
            step_success = np.asarray(info.get("is_success", np.zeros(len(seeds), dtype=bool)), dtype=bool)
            success |= step_success & active
            for i in np.flatnonzero(active):
                expert_action, distance = reference.query(states[i])
                base_error = float(np.linalg.norm(base_actions[i] - expert_action))
                adapted_error = float(np.linalg.norm(actions[i] - expert_action))
                state_ood = distance > reference.support_threshold
                harmful = bool(update and not state_ood and adapted_error > base_error + self.cfg.harmful_margin)
                logs[i].append(
                    {
                        "t": step,
                        "error": state_ood or adapted_error > reference.error_threshold,
                        "state_ood": state_ood,
                        "update_applied": update,
                        "harmful_update": harmful,
                        "proxy_norm": float(proxy[i].norm().item()),
                        "adaptive_state_norm": float(residual[i].norm().item()),
                        "adaptive_state_delta_norm": float((residual[i] - before[i]).norm().item()),
                        "action_delta_l2": float(np.linalg.norm(actions[i] - base_actions[i])),
                        "base_expert_error": base_error,
                        "adapted_expert_error": adapted_error,
                        "expert_nn_distance": distance,
                        "reward": float(reward[i]),
                    }
                )
            active &= ~(terminated | truncated)
            if not active.any():
                break
        return [
            {
                "suite": self.cfg.suite,
                "task_id": task_id,
                "task_description": description,
                "seed": seed,
                "condition": condition,
                "shift_id": self.cfg.shift_id,
                "shift": {
                    "camera_roll_deg": self.cfg.camera_roll_deg,
                    "object_dx_cm": 0.0,
                    "object_dy_cm": 0.0,
                    "action_bias": self.cfg.action_bias,
                    "gripper_delay_steps": self.cfg.gripper_delay_steps,
                },
                "success": bool(success[i]),
                "valid": True,
                "num_steps": len(logs[i]),
                "reference": reference.metadata,
                "probe": {
                    "update_interval": self.cfg.update_interval,
                    "eta": self.cfg.update_eta,
                    "decay": self.cfg.residual_decay,
                    "clip": self.cfg.residual_clip,
                    "proxy_roll_deg": self.cfg.proxy_roll_deg,
                },
                "batch_size": self.batch_size,
                "steps": logs[i],
            }
            for i, seed in enumerate(seeds)
        ]

    def run(self):
        output = Path(self.cfg.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = set()
        if output.exists():
            for line in output.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    completed.add((r["task_id"], r["seed"], r["condition"], r["shift_id"]))
        with output.open("a") as stream:
            for task_id in self.cfg.task_ids:
                for condition in self.cfg.conditions:
                    pending = [
                        seed
                        for seed in self.cfg.seeds
                        if (task_id, seed, condition, self.cfg.shift_id) not in completed
                    ]
                    for start in range(0, len(pending), self.batch_size):
                        actual = pending[start : start + self.batch_size]
                        # Fixed-size VectorEnv: pad the final group with duplicate seeds,
                        # then discard padded records.
                        padded = actual + [actual[-1]] * (self.batch_size - len(actual))
                        records = self.run_batch(task_id, padded, condition)[: len(actual)]
                        for record in records:
                            stream.write(json.dumps(record) + "\n")
                            stream.flush()
                            print(
                                f"task={task_id} seed={record['seed']} condition={condition} "
                                f"success={record['success']} steps={record['num_steps']}",
                                flush=True,
                            )


def _ints(value):
    return tuple(int(x) for x in value.split(",") if x)


def _strings(value):
    return tuple(x for x in value.split(",") if x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=_ints, default=(0,))
    parser.add_argument("--seeds", type=_ints, default=(0,))
    parser.add_argument("--conditions", type=_strings, default=CONDITIONS)
    parser.add_argument("--camera-roll-deg", type=float, default=0)
    parser.add_argument("--action-bias", type=float, default=0)
    parser.add_argument("--gripper-delay-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=3)
    args = vars(parser.parse_args())
    batch_size = args.pop("batch_size")
    cfg = ProbeConfig(**args)
    runner = BatchedProbeRunner(cfg, batch_size)
    try:
        runner.run()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
