from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditions = list(data["conditions"])
    nar = [data["conditions"][c]["paired_adaptation"]["nar"] for c in conditions]
    par = [data["conditions"][c]["paired_adaptation"]["par"] for c in conditions]
    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, nar, 0.4, label="NAR")
    ax.bar(x + 0.2, par, 0.4, label="PAR")
    ax.set_xticks(x, conditions, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Paired episode rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "nar_par.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for condition in conditions:
        drift = data["conditions"][condition]["drift_by_update_count"]
        counts = sorted(map(int, drift))
        if not counts:
            continue
        ax.plot(counts, [drift[str(i)]["mean_action_delta"] for i in counts], label=condition)
    ax.set_xlabel("Cumulative update count")
    ax.set_ylabel("Mean action residual L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "drift.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
