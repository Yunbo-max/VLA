from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION

from .reference import ExpertActionReference


CONDITIONS = ("frozen", "online_persistent", "online_reset", "buffer_offline")


@dataclass
class ProbeConfig:
    checkpoint: str
    dataset_root: str
    output: str
    suite: str = "libero_spatial"
    task_ids: tuple[int, ...] = (0,)
    seeds: tuple[int, ...] = (0,)
    conditions: tuple[str, ...] = CONDITIONS
    camera_roll_deg: float = 0.0
    object_dx_cm: float = 0.0
    object_dy_cm: float = 0.0
    action_bias: float = 0.0
    gripper_delay_steps: int = 0
    update_interval: int = 5
    update_eta: float = 0.35
    residual_decay: float = 0.995
    residual_clip: float = 0.25
    reset_horizon: int = 20
    proxy_roll_deg: float = 2.0
    harmful_margin: float = 0.02
    action_error_threshold: float = 0.35
    max_steps: int | None = None
    device: str = "cuda"

    @property
    def shift_id(self) -> str:
        return (
            f"cam{self.camera_roll_deg:g}_obj{self.object_dx_cm:g}_{self.object_dy_cm:g}"
            f"_act{self.action_bias:g}_grip{self.gripper_delay_steps}"
        )


def _rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    if degrees == 0:
        return image
    height, width = image.shape[-3:-1]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    if image.ndim == 4:
        return np.stack(
            [cv2.warpAffine(x, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101) for x in image]
        )
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)


def _camera_shift(observation: dict[str, Any], degrees: float) -> dict[str, Any]:
    shifted = copy.deepcopy(observation)
    if degrees:
        for key, image in shifted["pixels"].items():
            shifted["pixels"][key] = _rotate_image(image, degrees)
    return shifted


def _policy_batch(raw_observation, task_description, env_preprocessor, preprocessor):
    batch = preprocess_observation(raw_observation)
    first_value = next(iter(raw_observation["pixels"].values()))
    batch_size = int(first_value.shape[0])
    batch["task"] = [task_description] * batch_size
    batch = env_preprocessor(batch)
    expert_state = batch["observation.state"].detach().cpu().numpy().copy()
    if batch_size == 1:
        expert_state = expert_state[0]
    return preprocessor(batch), expert_state


def _noise(policy, seed: int, step: int) -> torch.Tensor:
    generator = torch.Generator(device=policy.config.device)
    generator.manual_seed(seed * 1_000_003 + step)
    return torch.randn(
        (1, policy.config.chunk_size, policy.config.max_action_dim),
        generator=generator,
        device=policy.config.device,
    )


def _normalized_action(policy, batch, noise):
    policy.reset()
    return policy.predict_action_chunk(batch, noise=noise)[:, 0, :]


def _to_env_action(action, postprocessor, env_postprocessor) -> np.ndarray:
    transition = {ACTION: postprocessor(action)}
    action = env_postprocessor(transition)[ACTION]
    return action.detach().cpu().numpy()[0].astype(np.float32)


def _set_init_state_id(vec_env, seed: int) -> None:
    base = vec_env.envs[0]
    base.init_state_id = seed % len(base._init_states)


def _apply_object_shift(vec_env, dx_cm: float, dy_cm: float) -> dict[str, Any] | None:
    if dx_cm == 0 and dy_cm == 0:
        return None
    base = vec_env.envs[0]
    model = base._env.env.sim.model
    data = base._env.env.sim.data
    bddl_text = Path(base._task_bddl_file).read_text()
    interest_match = re.search(r"\(:obj_of_interest\s+([^\)]+)\)", bddl_text, flags=re.DOTALL)
    target_object = interest_match.group(1).split()[0] if interest_match else None
    candidates = []
    for joint_id in range(model.njnt):
        # MuJoCo free joint = 0; robot joints are hinge/slide and therefore excluded.
        if int(model.jnt_type[joint_id]) == 0:
            name = model.joint_id2name(joint_id)
            address = int(model.jnt_qposadr[joint_id])
            candidates.append((name, address))
    if not candidates:
        raise RuntimeError("No free object joint found for object-pose shift")
    matching = [item for item in candidates if target_object and item[0].startswith(f"{target_object}_joint")]
    name, address = (matching or candidates)[0]
    before = data.qpos[address : address + 3].copy()
    qpos_snapshot = data.qpos.copy()
    qvel_snapshot = data.qvel.copy()
    requested = np.array([dx_cm, dy_cm], dtype=np.float64) / 100.0
    candidates_xy = [requested, -requested, requested[::-1] * np.array([1, -1]), requested[::-1] * np.array([-1, 1])]
    accepted = None
    raw_obs = None
    for candidate in candidates_xy:
        data.qpos[:] = qpos_snapshot
        data.qvel[:] = qvel_snapshot
        data.qpos[address : address + 2] += candidate
        base._env.env.sim.forward()
        for _ in range(3):
            raw_obs, _, _, _ = base._env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
        actual = data.qpos[address : address + 3].copy() - before
        # Reject collision-induced launches/falls. Allow 2 cm settling slack.
        if np.linalg.norm(actual[:2] - candidate) <= 0.02 and abs(actual[2]) <= 0.02:
            accepted = candidate
            break
    if accepted is None:
        data.qpos[:] = qpos_snapshot
        data.qvel[:] = qvel_snapshot
        base._env.env.sim.forward()
        raise RuntimeError(
            f"Could not apply a collision-stable {dx_cm:g},{dy_cm:g} cm shift to {name}; simulator restored"
        )
    formatted = base._format_raw_obs(raw_obs)
    return {
        "joint": name,
        "bddl_target_object": target_object,
        "requested_xy_m": requested.tolist(),
        "accepted_xy_m": accepted.tolist(),
        "before_xyz": before.tolist(),
        "after_xyz": data.qpos[address : address + 3].copy().tolist(),
        "observation": formatted,
    }


def _batch_observation(single: dict[str, Any]) -> dict[str, Any]:
    def add_batch(value):
        if isinstance(value, dict):
            return {key: add_batch(item) for key, item in value.items()}
        return np.expand_dims(value, 0)

    return add_batch(single)


class ProbeRunner:
    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        torch.backends.cuda.matmul.allow_tf32 = True
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
        self.envs = make_env(self.env_cfg, n_envs=1, use_async_envs=False)
        self.references: dict[int, ExpertActionReference] = {}

    def close(self):
        for env in self.envs[self.cfg.suite].values():
            env.close()

    def _reset(self, env, seed: int):
        _set_init_state_id(env, seed)
        observation, _ = env.reset(seed=[seed])
        object_info = _apply_object_shift(env, self.cfg.object_dx_cm, self.cfg.object_dy_cm)
        if object_info is not None:
            observation = _batch_observation(object_info.pop("observation"))
        return observation, object_info

    def _reference(self, task_id: int, description: str):
        if task_id not in self.references:
            self.references[task_id] = ExpertActionReference(self.cfg.dataset_root, description)
        return self.references[task_id]

    @torch.inference_mode()
    def _actions_and_proxy(self, observation, description, seed, step, need_proxy):
        shifted = _camera_shift(observation, self.cfg.camera_roll_deg)
        batch, state = _policy_batch(shifted, description, self.env_preprocessor, self.preprocessor)
        noise = _noise(self.policy, seed, step)
        base = _normalized_action(self.policy, batch, noise)
        proxy = torch.zeros_like(base)
        if need_proxy:
            augmented = _camera_shift(observation, self.cfg.camera_roll_deg + self.cfg.proxy_roll_deg)
            aug_batch, _ = _policy_batch(augmented, description, self.env_preprocessor, self.preprocessor)
            aug = _normalized_action(self.policy, aug_batch, noise)
            proxy = aug - base
        return base, proxy, state

    def _calibration_pass(self, env, seed, description):
        observation, _ = self._reset(env, seed)
        proxies = []
        max_steps = env.call("_max_episode_steps")[0]
        for step in range(max_steps):
            need_proxy = step % self.cfg.update_interval == 0
            base, proxy, _ = self._actions_and_proxy(observation, description, seed, step, need_proxy)
            if need_proxy:
                proxies.append(proxy.detach().clone())
            action = _to_env_action(base, self.postprocessor, self.env_postprocessor)
            observation, _, terminated, truncated, _ = env.step(action[None])
            if bool(terminated[0] or truncated[0]):
                break
        if not proxies:
            return torch.zeros((1, 7), device=self.cfg.device)
        residual = self.cfg.update_eta * torch.stack(proxies).mean(dim=0)
        return residual.clamp(-self.cfg.residual_clip, self.cfg.residual_clip)

    def run_episode(self, task_id: int, seed: int, condition: str) -> dict[str, Any]:
        if condition not in CONDITIONS:
            raise ValueError(condition)
        env = self.envs[self.cfg.suite][task_id]
        description = env.call("task_description")[0]
        reference = self._reference(task_id, description)
        fixed_residual = (
            self._calibration_pass(env, seed, description)
            if condition == "buffer_offline"
            else torch.zeros((1, 7), device=self.cfg.device)
        )
        observation, object_info = self._reset(env, seed)
        residual = fixed_residual.clone()
        delay = deque(maxlen=max(1, self.cfg.gripper_delay_steps + 1))
        steps = []
        max_steps = env.call("_max_episode_steps")[0]
        success = False
        for step in range(max_steps):
            if condition == "online_reset" and step and step % self.cfg.reset_horizon == 0:
                residual.zero_()
            update = condition in ("online_persistent", "online_reset") and step % self.cfg.update_interval == 0
            base_norm, proxy, state = self._actions_and_proxy(observation, description, seed, step, update)
            residual_before = residual.clone()
            if update:
                residual.mul_(self.cfg.residual_decay).add_(self.cfg.update_eta * proxy)
                residual.clamp_(-self.cfg.residual_clip, self.cfg.residual_clip)
            adapted_norm = base_norm if condition == "frozen" else base_norm + residual
            base_action = _to_env_action(base_norm, self.postprocessor, self.env_postprocessor)
            action = _to_env_action(adapted_norm, self.postprocessor, self.env_postprocessor)
            if self.cfg.action_bias:
                action[:3] = np.clip(action[:3] + self.cfg.action_bias, -1, 1)
            delay.append(float(action[6]))
            if self.cfg.gripper_delay_steps and len(delay) > self.cfg.gripper_delay_steps:
                action[6] = delay[0]
            expert_action, expert_distance = reference.query(state)
            base_error = float(np.linalg.norm(base_action - expert_action))
            adapted_error = float(np.linalg.norm(action - expert_action))
            state_ood = expert_distance > reference.support_threshold
            harmful = bool(update and not state_ood and adapted_error > base_error + self.cfg.harmful_margin)
            action_delta = float(np.linalg.norm(action - base_action))
            observation, reward, terminated, truncated, info = env.step(action[None])
            success = bool(info.get("is_success", np.array([False]))[0])
            steps.append(
                {
                    "t": step,
                    "error": state_ood or adapted_error > reference.error_threshold,
                    "state_ood": state_ood,
                    "update_applied": update,
                    "harmful_update": harmful,
                    "proxy_norm": float(proxy.norm().item()),
                    "adaptive_state_norm": float(residual.norm().item()),
                    "adaptive_state_delta_norm": float((residual - residual_before).norm().item()),
                    "action_delta_l2": action_delta,
                    "base_expert_error": base_error,
                    "adapted_expert_error": adapted_error,
                    "expert_nn_distance": expert_distance,
                    "reward": float(reward[0]),
                }
            )
            if bool(terminated[0] or truncated[0]):
                break
        return {
            "suite": self.cfg.suite,
            "task_id": task_id,
            "task_description": description,
            "seed": seed,
            "condition": condition,
            "shift_id": self.cfg.shift_id,
            "shift": {
                "camera_roll_deg": self.cfg.camera_roll_deg,
                "object_dx_cm": self.cfg.object_dx_cm,
                "object_dy_cm": self.cfg.object_dy_cm,
                "action_bias": self.cfg.action_bias,
                "gripper_delay_steps": self.cfg.gripper_delay_steps,
                "object_info": object_info,
            },
            "success": success,
            "num_steps": len(steps),
            "reference": reference.metadata,
            "probe": {
                "update_interval": self.cfg.update_interval,
                "eta": self.cfg.update_eta,
                "decay": self.cfg.residual_decay,
                "clip": self.cfg.residual_clip,
                "proxy_roll_deg": self.cfg.proxy_roll_deg,
            },
            "steps": steps,
        }

    def run(self):
        output = Path(self.cfg.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = set()
        if output.exists():
            for line in output.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    completed.add((record["task_id"], record["seed"], record["condition"], record["shift_id"]))
        with output.open("a") as stream:
            for task_id in self.cfg.task_ids:
                for seed in self.cfg.seeds:
                    for condition in self.cfg.conditions:
                        key = (task_id, seed, condition, self.cfg.shift_id)
                        if key in completed:
                            continue
                        try:
                            record = self.run_episode(task_id, seed, condition)
                            record["valid"] = True
                        except RuntimeError as exc:
                            record = {
                                "suite": self.cfg.suite,
                                "task_id": task_id,
                                "seed": seed,
                                "condition": condition,
                                "shift_id": self.cfg.shift_id,
                                "success": False,
                                "valid": False,
                                "invalid_reason": str(exc),
                                "steps": [],
                            }
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
                        print(
                            f"task={task_id} seed={seed} condition={condition} "
                            f"valid={record['valid']} success={record['success']} "
                            f"steps={record.get('num_steps', 0)}",
                            flush=True,
                        )


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x)


def _strings(value: str) -> tuple[str, ...]:
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
    parser.add_argument("--object-dx-cm", type=float, default=0)
    parser.add_argument("--object-dy-cm", type=float, default=0)
    parser.add_argument("--action-bias", type=float, default=0)
    parser.add_argument("--gripper-delay-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    cfg = ProbeConfig(**vars(args))
    runner = ProbeRunner(cfg)
    try:
        runner.run()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
