from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PAIR_FIELDS = ("suite", "task_id", "seed", "shift_id")


def pair_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in PAIR_FIELDS)


def paired_adaptation_rates(records: list[dict[str, Any]], condition: str) -> dict[str, float | int]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[pair_key(record)][record["condition"]] = record

    eligible_success = eligible_failure = negative = positive = 0
    for conditions in grouped.values():
        if "frozen" not in conditions or condition not in conditions:
            continue
        before = bool(conditions["frozen"]["success"])
        after = bool(conditions[condition]["success"])
        if before:
            eligible_success += 1
            negative += int(not after)
        else:
            eligible_failure += 1
            positive += int(after)
    return {
        "paired_n": eligible_success + eligible_failure,
        "frozen_success_n": eligible_success,
        "frozen_failure_n": eligible_failure,
        "negative_adaptation_n": negative,
        "positive_adaptation_n": positive,
        "nar": negative / eligible_success if eligible_success else float("nan"),
        "par": positive / eligible_failure if eligible_failure else float("nan"),
    }


def _paired_outcomes(records: list[dict[str, Any]], condition: str) -> list[tuple[bool, bool]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[pair_key(record)][record["condition"]] = record
    return [
        (bool(group["frozen"]["success"]), bool(group[condition]["success"]))
        for group in grouped.values()
        if "frozen" in group and condition in group
    ]


def paired_bootstrap_ci(
    records: list[dict[str, Any]], condition: str, samples: int = 10000, seed: int = 0
) -> dict[str, list[float]]:
    outcomes = _paired_outcomes(records, condition)
    if not outcomes:
        return {"nar_95ci": [float("nan"), float("nan")], "par_95ci": [float("nan"), float("nan")]}
    array = np.asarray(outcomes, dtype=bool)
    rng = np.random.default_rng(seed)
    nar, par = [], []
    for _ in range(samples):
        draw = array[rng.integers(0, len(array), len(array))]
        before, after = draw[:, 0], draw[:, 1]
        if before.any():
            nar.append(float((before & ~after).sum() / before.sum()))
        if (~before).any():
            par.append(float((~before & after).sum() / (~before).sum()))
    quantiles = lambda values: np.quantile(values, [0.025, 0.975]).tolist() if values else [float("nan")] * 2
    return {"nar_95ci": quantiles(nar), "par_95ci": quantiles(par)}


def propagation_risk(
    records: list[dict[str, Any]], condition: str, horizons: list[int], bootstrap_samples: int = 10000
) -> dict[str, Any]:
    selected = [r for r in records if r["condition"] == condition]
    unconditional = defaultdict(list)
    conditional = defaultdict(list)
    events = 0
    for record in selected:
        steps = record.get("steps", [])
        errors = [bool(step.get("error", False)) for step in steps]
        harmful = [i for i, step in enumerate(steps) if step.get("harmful_update", False)]
        for horizon in horizons:
            unconditional[horizon].extend(errors[horizon:])
        if not harmful:
            continue
        events += 1
        t0 = harmful[0]
        for horizon in horizons:
            target = t0 + horizon
            if target < len(errors):
                conditional[horizon].append(errors[target])

    output: dict[str, Any] = {"first_harmful_update_events": events, "horizons": {}}
    for horizon in horizons:
        base = sum(unconditional[horizon]) / len(unconditional[horizon]) if unconditional[horizon] else float("nan")
        cond = sum(conditional[horizon]) / len(conditional[horizon]) if conditional[horizon] else float("nan")
        output["horizons"][str(horizon)] = {
            "unconditional_error_probability": base,
            "conditional_error_probability": cond,
            "risk_ratio": cond / base if base and base == base else float("nan"),
            "conditional_n": len(conditional[horizon]),
        }
        # Episode-cluster bootstrap: resample whole trajectories so temporal
        # correlation inside an episode is never treated as independent evidence.
        if selected and bootstrap_samples:
            per_episode = []
            for record in selected:
                errors = np.asarray([bool(step.get("error", False)) for step in record.get("steps", [])])
                harmful = [i for i, step in enumerate(record.get("steps", [])) if step.get("harmful_update", False)]
                unconditional_slice = errors[horizon:]
                conditional_value = None
                if harmful and harmful[0] + horizon < len(errors):
                    conditional_value = bool(errors[harmful[0] + horizon])
                per_episode.append(
                    (
                        int(unconditional_slice.sum()),
                        int(unconditional_slice.size),
                        int(conditional_value) if conditional_value is not None else 0,
                        int(conditional_value is not None),
                    )
                )
            values = np.asarray(per_episode, dtype=np.float64)
            rng = np.random.default_rng(7919 + horizon)
            ratios = []
            for _ in range(bootstrap_samples):
                draw = values[rng.integers(0, len(values), len(values))].sum(axis=0)
                if draw[1] and draw[3] and draw[0]:
                    ratios.append((draw[2] / draw[3]) / (draw[0] / draw[1]))
            output["horizons"][str(horizon)]["risk_ratio_95ci"] = (
                np.quantile(ratios, [0.025, 0.975]).tolist() if ratios else [float("nan"), float("nan")]
            )
    return output


def update_necessity(records: list[dict[str, Any]], condition: str, threshold: float) -> dict[str, float | int]:
    steps = [step for r in records if r["condition"] == condition for step in r.get("steps", [])]
    changed = [step for step in steps if float(step.get("action_delta_l2", 0.0)) > threshold]
    harmful = sum(bool(step.get("harmful_update", False)) for step in changed)
    return {
        "decision_n": len(steps),
        "changed_decision_n": len(changed),
        "changed_fraction": len(changed) / len(steps) if steps else float("nan"),
        "harmful_changed_n": harmful,
        "harmful_among_changed": harmful / len(changed) if changed else float("nan"),
    }


def drift_summary(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    by_update: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["condition"] != condition:
            continue
        update_count = 0
        for step in record.get("steps", []):
            update_count += int(step.get("update_applied", False))
            by_update[update_count].append(step)
    return {
        str(count): {
            "n": len(steps),
            "mean_state_norm": float(np.mean([s.get("adaptive_state_norm", 0.0) for s in steps])),
            "mean_action_delta": float(np.mean([s.get("action_delta_l2", 0.0) for s in steps])),
            "mean_expert_error": float(np.mean([s.get("adapted_expert_error", np.nan) for s in steps])),
        }
        for count, steps in sorted(by_update.items())
    }


def online_buffer_gap(records: list[dict[str, Any]]) -> dict[str, float | int]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[pair_key(record)][record["condition"]] = record
    pairs = [g for g in grouped.values() if "online_persistent" in g and "buffer_offline" in g]
    if not pairs:
        return {"paired_n": 0, "online_minus_buffer_success": float("nan")}
    differences = [
        int(g["online_persistent"]["success"]) - int(g["buffer_offline"]["success"]) for g in pairs
    ]
    return {"paired_n": len(pairs), "online_minus_buffer_success": float(np.mean(differences))}


def compute(
    records: list[dict[str, Any]], horizons: list[int], threshold: float, bootstrap_samples: int = 10000
) -> dict[str, Any]:
    invalid_n = sum(not record.get("valid", True) for record in records)
    records = [record for record in records if record.get("valid", True)]
    conditions = sorted({r["condition"] for r in records} - {"frozen"})
    return {
        "record_n": len(records),
        "invalid_record_n": invalid_n,
        "conditions": {
            condition: {
                "paired_adaptation": paired_adaptation_rates(records, condition),
                "paired_bootstrap": paired_bootstrap_ci(records, condition, bootstrap_samples),
                "propagation": propagation_risk(records, condition, horizons, bootstrap_samples),
                "update_necessity": update_necessity(records, condition, threshold),
                "drift_by_update_count": drift_summary(records, condition),
            }
            for condition in conditions
        },
        "online_buffer_gap": online_buffer_gap(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--action-threshold", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    result = compute(records, args.horizons, args.action_threshold, args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
