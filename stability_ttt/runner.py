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
ORACLE_CONDITION = "oracle_selective_commit"


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


def _noise_batch(policy, seeds: list[int], step: int) -> torch.Tensor:
    return torch.cat([_noise(policy, seed, step) for seed in seeds], dim=0)


def _normalized_action(policy, batch, noise):
    policy.reset()
    return policy.predict_action_chunk(batch, noise=noise)[:, 0, :]


def _to_env_action(action, postprocessor, env_postprocessor) -> np.ndarray:
    transition = {ACTION: postprocessor(action)}
    action = env_postprocessor(transition)[ACTION]
    return action.detach().cpu().numpy()[0].astype(np.float32)


def _to_env_actions(action, postprocessor, env_postprocessor) -> np.ndarray:
    transition = {ACTION: postprocessor(action)}
    action = env_postprocessor(transition)[ACTION]
    return action.detach().cpu().numpy().astype(np.float32)


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


def _stack_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack two raw formatted observations for matched branch inference."""
    first = observations[0]

    def stack(values):
        if isinstance(values[0], dict):
            return {key: stack([value[key] for value in values]) for key in values[0]}
        return np.concatenate([np.asarray(value)[None, ...] for value in values], axis=0)

    return stack(observations)


def _select_observation(batch: dict[str, Any], index: int) -> dict[str, Any]:
    """Select one batch element while retaining its leading batch dimension."""
    def select(value):
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        return np.asarray(value)[index : index + 1]

    return select(batch)


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
        self.oracle_envs = None
        if ORACLE_CONDITION in cfg.conditions:
            # Two simulator instances are used only for the commit / rollback
            # probe. Their observations are batched into one model forward,
            # while each simulator remains independent.
            self.oracle_envs = make_env(self.env_cfg, n_envs=2, use_async_envs=False)
        self.references: dict[int, ExpertActionReference] = {}

    def close(self):
        for env in self.envs[self.cfg.suite].values():
            env.close()
        if self.oracle_envs is not None:
            for env in self.oracle_envs[self.cfg.suite].values():
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

    @staticmethod
    def _runtime_snapshot(env):
        """Capture simulator and task-clock state for matched counterfactuals."""
        task = env.envs[0]._env.env
        return {
            "sim_state": np.array(env.envs[0]._env.get_sim_state(), copy=True),
            "timestep": int(task.timestep),
            "cur_time": float(task.cur_time),
            "done": bool(task.done),
        }

    @staticmethod
    def _restore_runtime(env, snapshot):
        ProbeRunner._restore_base_runtime(env.envs[0], snapshot)

    @staticmethod
    def _restore_base_runtime(base, snapshot):
        task = base._env.env
        base._env.set_state(np.array(snapshot["sim_state"], copy=True))
        task.timestep = snapshot["timestep"]
        task.cur_time = snapshot["cur_time"]
        task.done = snapshot["done"]

    @staticmethod
    def _success_from_info(info) -> bool:
        value = info.get("is_success", False) if isinstance(info, dict) else False
        array = np.asarray(value).reshape(-1)
        return bool(array[0]) if array.size else bool(value)

    def _oracle_single_branch(
        self,
        branch_env,
        branch_index: int,
        description: str,
        seed: int,
        start_step: int,
        snapshot: dict[str, Any],
        residual: torch.Tensor,
        delay_values: list[float],
    ) -> bool:
        """Roll one matched branch from an exact state with adaptation frozen.

        The branch uses only the supplied fixed residual. It never computes or
        applies a new proxy update, so the sole manipulated variable is whether
        the current tentative update persists.
        """
        base = branch_env.envs[branch_index]
        for branch_base in branch_env.envs:
            self._restore_base_runtime(branch_base, snapshot)
        raw = base._env.regenerate_obs_from_state(snapshot["sim_state"])
        observation = _batch_observation(base._format_raw_obs(raw))
        delay = deque(delay_values, maxlen=max(1, self.cfg.gripper_delay_steps + 1))
        success = False
        max_steps = branch_env.call("_max_episode_steps")[0]
        for branch_step in range(start_step, max_steps):
            shifted = _camera_shift(observation, self.cfg.camera_roll_deg)
            batch, _ = _policy_batch(shifted, description, self.env_preprocessor, self.preprocessor)
            base_norm = _normalized_action(self.policy, batch, _noise(self.policy, seed, branch_step))
            adapted_norm = base_norm + residual
            actions = _to_env_actions(adapted_norm, self.postprocessor, self.env_postprocessor)
            if self.cfg.action_bias:
                actions[0, :3] = np.clip(actions[0, :3] + self.cfg.action_bias, -1, 1)
            delay.append(float(actions[0, 6]))
            if self.cfg.gripper_delay_steps and len(delay) > self.cfg.gripper_delay_steps:
                actions[0, 6] = delay[0]
            step_actions = np.zeros((2, actions.shape[1]), dtype=np.float32)
            step_actions[branch_index] = actions[0]
            # SyncVectorEnv steps both workers. Keep the non-target worker
            # alive and irrelevant so a prior dummy step cannot make the next
            # vector step fail with "executing action in terminated episode".
            self._restore_base_runtime(branch_env.envs[1 - branch_index], snapshot)
            observations, _, terminated, truncated, info = branch_env.step(step_actions)
            observation = _select_observation(observations, branch_index)
            values = info.get("is_success", False) if isinstance(info, dict) else False
            success_values = np.asarray(values, dtype=bool).reshape(-1)
            success = bool(success_values[branch_index])
            term_values = np.asarray(terminated).reshape(-1)
            trunc_values = np.asarray(truncated).reshape(-1)
            if success or bool(term_values[branch_index] or trunc_values[branch_index]):
                break
        return success

    def _oracle_branch(
        self,
        env,
        task_id: int,
        description: str,
        seed: int,
        start_step: int,
        snapshot: dict[str, Any],
        rollback_residual: torch.Tensor,
        commit_residual: torch.Tensor,
        delay_values: list[float],
    ) -> tuple[bool, bool]:
        """Evaluate U exactly, short-circuiting when commit already fails.

        Since terminal success is binary, a failed commit implies
        ``G_commit - G_rollback <= 0`` regardless of the rollback branch.  The
        rollback branch is therefore only simulated when commit succeeds; this
        is an exact optimization, not a changed oracle criterion.
        """
        if self.oracle_envs is None:
            raise RuntimeError("oracle branch environments were not initialized")
        branch_env = self.oracle_envs[self.cfg.suite][task_id]
        if branch_env.envs[0]._env is None:
            branch_env.reset(seed=[seed, seed])
        commit_success = self._oracle_single_branch(
            branch_env, 1, description, seed, start_step, snapshot,
            commit_residual, delay_values,
        )
        if not commit_success:
            return False, False
        rollback_success = self._oracle_single_branch(
            branch_env, 0, description, seed, start_step, snapshot,
            rollback_residual, delay_values,
        )
        return True, rollback_success

    def run_episode(self, task_id: int, seed: int, condition: str) -> dict[str, Any]:
        if condition not in CONDITIONS and condition != ORACLE_CONDITION:
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
            update = (
                condition in ("online_persistent", "online_reset", ORACLE_CONDITION)
                and step % self.cfg.update_interval == 0
            )
            base_norm, proxy, state = self._actions_and_proxy(observation, description, seed, step, update)
            residual_before = residual.clone()
            tentative = residual_before.clone()
            if update:
                tentative = (
                    residual_before * self.cfg.residual_decay + self.cfg.update_eta * proxy
                ).clamp(-self.cfg.residual_clip, self.cfg.residual_clip)
                if condition != ORACLE_CONDITION:
                    residual = tentative.clone()
            adapted_norm = base_norm if condition == "frozen" else base_norm + residual
            if condition == ORACLE_CONDITION:
                # The current action uses the tentative update; persistence is
                # decided only after observing its consequence.
                adapted_norm = base_norm + tentative
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
            delay_values = list(delay)
            observation, reward, terminated, truncated, info = env.step(action[None])
            success = success or self._success_from_info(info)
            oracle_commit = False
            oracle_g_commit = None
            oracle_g_rollback = None
            oracle_utility = None
            if condition == ORACLE_CONDITION and update and not bool(terminated[0] or truncated[0]):
                snapshot = self._runtime_snapshot(env)
                oracle_g_commit, oracle_g_rollback = self._oracle_branch(
                    env,
                    task_id,
                    description,
                    seed,
                    step + 1,
                    snapshot,
                    residual_before,
                    tentative,
                    delay_values,
                )
                self._restore_runtime(env, snapshot)
                oracle_utility = int(oracle_g_commit) - int(oracle_g_rollback)
                oracle_commit = oracle_utility > 0
                residual = tentative.clone() if oracle_commit else residual_before.clone()
            elif condition == ORACLE_CONDITION and update:
                # No future remains; U=0 by definition, so do not persist an
                # update that cannot affect a subsequent action.
                oracle_g_commit = bool(success)
                oracle_g_rollback = bool(success)
                oracle_utility = 0
                residual = residual_before.clone()
            if condition == ORACLE_CONDITION and not update:
                residual = residual_before.clone()
            steps.append(
                {
                    "t": step,
                    "error": state_ood or adapted_error > reference.error_threshold,
                    "state_ood": state_ood,
                    "update_applied": update,
                    "update_committed": oracle_commit if condition == ORACLE_CONDITION else update,
                    "harmful_update": harmful,
                    "proxy_norm": float(proxy.norm().item()),
                    "adaptive_state_norm": float(residual.norm().item()),
                    "adaptive_state_delta_norm": float((residual - residual_before).norm().item()),
                    "tentative_state_norm": float(tentative.norm().item()),
                    "tentative_update_norm": float((tentative - residual_before).norm().item()),
                    "oracle_g_commit": oracle_g_commit,
                    "oracle_g_rollback": oracle_g_rollback,
                    "consolidation_utility": oracle_utility,
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
                "oracle": condition == ORACLE_CONDITION,
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
