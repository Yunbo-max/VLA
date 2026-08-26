#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export MUJOCO_GL=egl
export HF_HOME="$project_root/.hf-cache"
export PYTHONPATH="$project_root"

# The shift argument is selected after scripts/screen_shifts.sh. Example:
#   scripts/run_pilot.sh --camera-roll-deg 10
.venv/bin/python -m stability_ttt.batch_runner \
  --checkpoint checkpoints/smolvla_libero \
  --dataset-root data/libero_lerobot \
  --output results/pilot.jsonl \
  --suite libero_spatial \
  --task-ids 0,1,2 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --conditions frozen,online_persistent,online_reset,buffer_offline \
  --batch-size 3 \
  "$@"
