#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root"
records="${1:-results/pilot.jsonl}"
name="$(basename "$records" .jsonl)"
.venv/bin/python -m stability_ttt.metrics \
  --records "$records" \
  --output "results/${name}_metrics.json" \
  --bootstrap-samples 10000
.venv/bin/python -m stability_ttt.plot \
  --metrics "results/${name}_metrics.json" \
  --output-dir "results/${name}_plots"
