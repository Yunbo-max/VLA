#!/usr/bin/env python3
"""Seed the main JSONL with compatible pilot records, preserving unique run keys."""

import argparse
import json
from pathlib import Path


def key(record: dict) -> tuple:
    return (
        record["suite"],
        record["task_id"],
        record["seed"],
        record["condition"],
        record["shift_id"],
    )


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    args = parser.parse_args()

    records: dict[tuple, dict] = {}
    for record in load(args.pilot) + load(args.main):
        if record.get("batch_size") != 3:
            raise ValueError("Only batch_size=3 records are compatible with the main design")
        if record.get("shift_id") != "cam10_obj0_0_act0_grip0":
            raise ValueError(f"Unexpected shift: {record.get('shift_id')}")
        records.setdefault(key(record), record)

    ordered = sorted(records.values(), key=lambda r: (r["task_id"], r["condition"], r["seed"]))
    args.main.write_text("".join(json.dumps(r, allow_nan=True) + "\n" for r in ordered))
    print(f"wrote {len(ordered)} unique compatible records to {args.main}")


if __name__ == "__main__":
    main()
