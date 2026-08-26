from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for record in records:
        grouped[record["shift_id"]].append(record)
    rows = []
    for shift_id, group in grouped.items():
        valid = [r for r in group if r.get("valid", True)]
        rows.append(
            {
                "shift_id": shift_id,
                "records": len(group),
                "valid": len(valid),
                "successes": sum(bool(r["success"]) for r in valid),
                "success_rate": sum(bool(r["success"]) for r in valid) / len(valid) if valid else None,
                "shift": valid[0].get("shift") if valid else None,
            }
        )
    rows.sort(key=lambda row: (row["success_rate"] is None, row["success_rate"] or 0, row["shift_id"]))
    complete = [row for row in rows if row["valid"] >= 9]
    selected = complete[0] if complete else None
    output = {"selected": selected, "criterion": "lowest frozen success among shifts with >=9 valid pairs", "shifts": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
