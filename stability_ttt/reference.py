from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from scipy.spatial import cKDTree


class ExpertActionReference:
    """State-nearest expert action reference used only for retrospective labels.

    The reference is never used to choose or update an executed action.  State
    dimensions are standardized task-wise before nearest-neighbour lookup.
    """

    def __init__(self, dataset_root: str | Path, task_description: str, k: int = 8):
        self.root = Path(dataset_root)
        self.k = k
        task_table = ds.dataset(self.root / "meta" / "tasks.parquet", format="parquet").to_table()
        descriptions = task_table.column_names
        # LeRobot writes task strings as the parquet index, so pandas is the reliable reader here.
        import pandas as pd

        tasks = pd.read_parquet(self.root / "meta" / "tasks.parquet")
        if task_description not in tasks.index:
            available = list(tasks.index)
            raise KeyError(f"Task description not in expert dataset: {task_description!r}; examples={available[:3]}")
        task_index = int(tasks.loc[task_description, "task_index"])

        dataset = ds.dataset(self.root / "data", format="parquet", partitioning="hive")
        table = dataset.to_table(
            columns=["observation.state", "action"],
            filter=ds.field("task_index") == task_index,
        )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if states.size == 0:
            raise RuntimeError(f"No expert frames for task_index={task_index}")
        self.mean = states.mean(axis=0)
        self.scale = states.std(axis=0).clip(min=1e-4)
        self.states = (states - self.mean) / self.scale
        self.actions = actions
        self.tree = cKDTree(self.states)
        calibration_k = min(self.k + 1, len(self.states))
        calibration_distances, calibration_indices = self.tree.query(self.states, k=calibration_k)
        # Column zero is the sample itself. Remaining neighbours estimate normal
        # expert action ambiguity at nearby states for this particular task.
        neighbor_distances = calibration_distances[:, 1:]
        neighbor_indices = calibration_indices[:, 1:]
        weights = 1.0 / (neighbor_distances + 1e-3)
        predicted = (self.actions[neighbor_indices] * weights[..., None]).sum(axis=1) / weights.sum(
            axis=1, keepdims=True
        )
        loo_errors = np.linalg.norm(predicted - self.actions, axis=1)
        self.error_threshold = float(np.quantile(loo_errors, 0.90))
        self.support_threshold = float(np.quantile(calibration_distances[:, 1], 0.99))
        self.metadata = {
            "task_description": task_description,
            "task_index": task_index,
            "frames": int(len(states)),
            "neighbors": k,
            "error_threshold_p90": self.error_threshold,
            "state_support_distance_p99": self.support_threshold,
            "calibration": "leave-one-out expert nearest-neighbor action error",
        }

    def query(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        normalized = (np.asarray(state, dtype=np.float32) - self.mean) / self.scale
        distances, indices = self.tree.query(normalized, k=min(self.k, len(self.states)))
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        weights = 1.0 / (distances + 1e-3)
        action = np.average(self.actions[indices], axis=0, weights=weights)
        return action.astype(np.float32), float(distances[0])
