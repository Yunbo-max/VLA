#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export MUJOCO_GL=egl
export HF_HOME="$project_root/.hf-cache"
export PYTHONPATH="$project_root"

common=(
  --checkpoint checkpoints/smolvla_libero
  --dataset-root data/libero_lerobot
  --output results/shift_screen.jsonl
  --suite libero_spatial
  --task-ids 0,1,2
  --seeds 0,1,2
  --conditions frozen
)

.venv/bin/python -m stability_ttt.runner "${common[@]}"
for value in 2 5 10; do
  .venv/bin/python -m stability_ttt.runner "${common[@]}" --camera-roll-deg "$value"
done
for value in 1 2 4; do
  .venv/bin/python -m stability_ttt.runner "${common[@]}" --object-dx-cm "$value"
done
for value in 0.005 0.01 0.02; do
  .venv/bin/python -m stability_ttt.runner "${common[@]}" --action-bias "$value"
done
for value in 1 2; do
  .venv/bin/python -m stability_ttt.runner "${common[@]}" --gripper-delay-steps "$value"
done

