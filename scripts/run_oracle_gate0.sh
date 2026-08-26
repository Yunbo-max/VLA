#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export MUJOCO_GL=egl
export HF_HOME="$project_root/.hf-cache"
export PYTHONPATH="$project_root"

camera_roll_deg="${1:-10}"
seeds="0,1,2"
tasks="0,1,2,3,4,5,6,7,8,9"

for suite in libero_spatial libero_object libero_goal; do
  .venv/bin/python -m stability_ttt.runner \
    --checkpoint checkpoints/smolvla_libero \
    --dataset-root data/libero_lerobot \
    --output "results/oracle_gate0_${suite}.jsonl" \
    --suite "$suite" \
    --task-ids "$tasks" \
    --seeds "$seeds" \
    --conditions oracle_selective_commit \
    --camera-roll-deg "$camera_roll_deg"
done
