#!/usr/bin/env python3
"""Analyze adaptation versus persistent consolidation.

This is deliberately post-hoc: the frozen/reference trajectory is used only to
classify rescue (frozen failure -> adapted success) and corruption (frozen
success -> adapted failure). It never drives an update.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(r: dict[str, Any]) -> tuple[Any, ...]:
    return (r["suite"], r["task_id"], r["seed"], r["shift_id"])


def outcome_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        grouped[key(r)][r["condition"]] = r
    out: dict[str, Any] = {}
    for condition in ("online_persistent", "online_reset", "buffer_offline"):
        rows = [g for g in grouped.values() if "frozen" in g and condition in g]
        corruption = sum(bool(g["frozen"]["success"]) and not bool(g[condition]["success"]) for g in rows)
        rescue = sum(not bool(g["frozen"]["success"]) and bool(g[condition]["success"]) for g in rows)
        frozen_success = sum(bool(g["frozen"]["success"]) for g in rows)
        frozen_failure = len(rows) - frozen_success
        out[condition] = {
            "paired_n": len(rows),
            "frozen_success_n": frozen_success,
            "frozen_failure_n": frozen_failure,
            "corruption_n": corruption,
            "corruption_rate": corruption / frozen_success if frozen_success else float("nan"),
            "rescue_n": rescue,
            "rescue_rate": rescue / frozen_failure if frozen_failure else float("nan"),
        }
    # Simple Fisher exact comparison on frozen-success episodes. This is the
    # same descriptive 2x2 test used in the diagnostic discussion; paired
    # bootstrap CIs remain the primary uncertainty statement.
    tests = {}
    for other in ("online_reset", "buffer_offline"):
        g_p = [g for g in grouped.values() if "frozen" in g and "online_persistent" in g and g["frozen"]["success"]]
        g_o = [g for g in grouped.values() if "frozen" in g and other in g and g["frozen"]["success"]]
        cp = sum(not bool(g["online_persistent"]["success"]) for g in g_p)
        co = sum(not bool(g[other]["success"]) for g in g_o)
        table = [[cp, len(g_p) - cp], [co, len(g_o) - co]]
        odds, pvalue = fisher_exact(table)
        tests[f"persistent_vs_{other.removeprefix('online_')}_on_frozen_success"] = {
            "table": table,
            "odds_ratio": float(odds),
            "p_value": float(pvalue),
            "note": "descriptive 2x2 Fisher test; paired bootstrap CIs are primary",
        }
    out["fisher_tests_on_frozen_success"] = tests
    return out


def accumulation(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for condition in ("online_persistent", "online_reset", "buffer_offline"):
        bins: dict[int, list[dict[str, float]]] = defaultdict(list)
        for r in records:
            if r["condition"] != condition:
                continue
            count = 0
            for step in r.get("steps", []):
                count += int(step.get("update_applied", False))
                bins[count].append(
                    {
                        "state": float(step.get("adaptive_state_norm", 0.0)),
                        "action": float(step.get("action_delta_l2", 0.0)),
                        "harmful": float(bool(step.get("harmful_update", False))),
                    }
                )
        rows = []
        for count, values in sorted(bins.items()):
            rows.append({
                "update_count": count,
                "n": len(values),
                "mean_state_norm": float(np.mean([v["state"] for v in values])),
                "mean_action_delta": float(np.mean([v["action"] for v in values])),
                "harmful_rate": float(np.mean([v["harmful"] for v in values])),
            })
        out[condition] = rows
    return out


def plot(outcomes: dict[str, Any], accum: dict[str, list[dict[str, Any]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["persistent", "reset", "buffer"]
    conditions = ["online_persistent", "online_reset", "buffer_offline"]
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, [outcomes[c]["corruption_rate"] for c in conditions], width, label="corruption")
    axes[0].bar(x + width / 2, [outcomes[c]["rescue_rate"] for c in conditions], width, label="rescue")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("paired episode rate")
    axes[0].set_title("Adaptation vs. consolidation")
    axes[0].legend(frameon=False)
    for c, label in zip(conditions[:2], labels[:2]):
        rows = accum[c]
        axes[1].plot([r["update_count"] for r in rows], [r["mean_state_norm"] for r in rows], label=f"state · {label}")
        axes[1].plot([r["update_count"] for r in rows], [r["mean_action_delta"] for r in rows], linestyle="--", label=f"action · {label}")
    axes[1].set_xlabel("cumulative update count")
    axes[1].set_ylabel("mean magnitude")
    axes[1].set_title("Accumulated adaptive drift")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plot", type=Path, required=True)
    args = ap.parse_args()
    records = load(args.records)
    result = {"record_n": len(records), "outcomes": outcome_table(records), "accumulation": accumulation(records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    plot(result["outcomes"], result["accumulation"], args.plot)


if __name__ == "__main__":
    main()
