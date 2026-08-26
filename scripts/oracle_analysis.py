#!/usr/bin/env python3
"""Gate 1 analysis: oracle selective consolidation versus frozen D0 baselines."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(r: dict[str, Any]) -> tuple[Any, ...]:
    return (r["suite"], r["task_id"], r["seed"], r["shift_id"])


def paired_nar(records: list[dict[str, Any]], condition: str) -> tuple[int, int, int, int]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        grouped[key(r)][r["condition"]] = r
    success = failure = negative = positive = 0
    for g in grouped.values():
        if "frozen" not in g or condition not in g:
            continue
        if bool(g["frozen"]["success"]):
            success += 1
            negative += int(not bool(g[condition]["success"]))
        else:
            failure += 1
            positive += int(bool(g[condition]["success"]))
    return success, failure, negative, positive


def task_cluster_bootstrap(
    records: list[dict[str, Any]], condition: str, samples: int = 10000, seed: int = 1729
) -> dict[str, Any]:
    grouped: dict[int, list[tuple[bool, bool]]] = defaultdict(list)
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        by_key[key(r)][r["condition"]] = r
    # task_id is in the outer record key; recover it without relying on a
    # duplicated field in every group.
    grouped = defaultdict(list)
    for k, g in by_key.items():
        if "frozen" in g and condition in g:
            grouped[int(k[1])].append((bool(g["frozen"]["success"]), bool(g[condition]["success"])))
    task_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    nar_values, par_values, delta_values = [], [], []
    for _ in range(samples):
        draw_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
        outcomes = [pair for task in draw_tasks for pair in grouped[int(task)]]
        before = np.asarray([p[0] for p in outcomes], dtype=bool)
        after = np.asarray([p[1] for p in outcomes], dtype=bool)
        nar = float((before & ~after).sum() / before.sum()) if before.any() else np.nan
        par = float((~before & after).sum() / (~before).sum()) if (~before).any() else np.nan
        nar_values.append(nar)
        par_values.append(par)
        delta_values.append(nar)
    # Point estimates are computed from the unresampled paired data.
    success, failure, negative, positive = paired_nar(records, condition)
    return {
        "task_n": len(task_ids),
        "task_ids": task_ids,
        "paired_n": success + failure,
        "frozen_success_n": success,
        "frozen_failure_n": failure,
        "negative_adaptation_n": negative,
        "positive_adaptation_n": positive,
        "nar": negative / success if success else np.nan,
        "par": positive / failure if failure else np.nan,
        "nar_95ci_task_cluster": np.nanquantile(nar_values, [0.025, 0.975]).tolist(),
        "par_95ci_task_cluster": np.nanquantile(par_values, [0.025, 0.975]).tolist(),
    }


def delta_nar_bootstrap(
    records: list[dict[str, Any]], left: str, right: str, samples: int = 10000, seed: int = 1731
) -> dict[str, Any]:
    grouped: dict[int, list[tuple[bool, bool, bool]]] = defaultdict(list)
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        by_key[key(r)][r["condition"]] = r
    for k, g in by_key.items():
        if "frozen" in g and left in g and right in g:
            # Restrict NAR comparison to episodes frozen-success under the
            # paired design; this is the estimand used in D0.
            if bool(g["frozen"]["success"]):
                grouped[int(k[1])].append((True, bool(g[left]["success"]), bool(g[right]["success"])))
    task_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        draw_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
        pairs = [p for task in draw_tasks for p in grouped[int(task)]]
        left_n = sum(not p[1] for p in pairs)
        right_n = sum(not p[2] for p in pairs)
        denom = len(pairs)
        values.append((left_n - right_n) / denom if denom else np.nan)
    pairs = [p for task in task_ids for p in grouped[task]]
    point = (sum(not p[1] for p in pairs) - sum(not p[2] for p in pairs)) / len(pairs)
    return {
        "task_n": len(task_ids),
        "paired_frozen_success_n": len(pairs),
        "delta_nar_left_minus_right": point,
        "ci_95_task_cluster": np.nanquantile(values, [0.025, 0.975]).tolist(),
    }


def oracle_utility(records: list[dict[str, Any]]) -> dict[str, Any]:
    proposed = committed = negative = positive = zero = 0
    utilities: list[int] = []
    by_count: dict[int, list[dict[str, float]]] = defaultdict(list)
    for r in records:
        count = 0
        for step in r.get("steps", []):
            if step.get("update_applied"):
                proposed += 1
                utility = step.get("consolidation_utility")
                if utility is not None:
                    utility = int(utility)
                    utilities.append(utility)
                    positive += int(utility > 0)
                    negative += int(utility < 0)
                    zero += int(utility == 0)
                committed += int(step.get("update_committed", False))
            count += int(step.get("update_committed", False))
            by_count[count].append(
                {
                    "state": float(step.get("adaptive_state_norm", 0.0)),
                    "action": float(step.get("action_delta_l2", 0.0)),
                }
            )
    drift = []
    for count, rows in sorted(by_count.items()):
        drift.append(
            {
                "committed_update_count": count,
                "n": len(rows),
                "mean_state_norm": float(np.mean([x["state"] for x in rows])),
                "mean_action_delta": float(np.mean([x["action"] for x in rows])),
            }
        )
    return {
        "proposed_updates": proposed,
        "committed_updates": committed,
        "commit_rate": committed / proposed if proposed else np.nan,
        "consolidation_harm_n": negative,
        "consolidation_harm_rate": negative / len(utilities) if utilities else np.nan,
        "utility_positive_n": positive,
        "utility_zero_n": zero,
        "utility_negative_n": negative,
        "utility_distribution": utilities,
        "drift_by_committed_update_count": drift,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    args = ap.parse_args()
    oracle = load(args.oracle)
    baseline = load(args.baseline)
    oracle_metrics = paired_nar(baseline + oracle, "oracle_selective_commit")
    success, failure, negative, positive = oracle_metrics
    result = {
        "oracle_record_n": len(oracle),
        "overall_success_rate": float(np.mean([bool(r["success"]) for r in oracle])) if oracle else np.nan,
        "paired": {
            "paired_n": success + failure,
            "frozen_success_n": success,
            "frozen_failure_n": failure,
            "negative_adaptation_n": negative,
            "positive_adaptation_n": positive,
            "nar": negative / success if success else np.nan,
            "par": positive / failure if failure else np.nan,
        },
        "task_cluster_bootstrap": task_cluster_bootstrap(
            baseline + oracle, "oracle_selective_commit", args.bootstrap_samples
        ),
        "delta_nar_persistent_minus_reset": delta_nar_bootstrap(
            baseline, "online_persistent", "online_reset", args.bootstrap_samples
        ),
        "delta_nar_oracle_minus_persistent": delta_nar_bootstrap(
            baseline + oracle, "oracle_selective_commit", "online_persistent", args.bootstrap_samples
        ),
        "oracle": oracle_utility(oracle),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
