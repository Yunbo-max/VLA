# Closed-Loop VLA Test-Time Learning Stability

Diagnostic-first experiments for the hypothesis that persistent online test-time updates can turn local VLA errors into closed-loop adaptation drift.

## Hardware-selected stack

- GPU: RTX 3080 10 GB
- Primary policy: `HuggingFaceVLA/smolvla_libero` (605M parameters, 1.2 GB checkpoint)
- Simulator/data: LIBERO through LeRobot / `hf-libero`
- Paper reference implementation: `third_party/CVPR2026-PDF`
- Paper-scale OpenVLA-7B is configuration-only on this host: FP16 inference plus adaptation does not fit safely in 10 GB.

The simulator contains task definitions and fixed initial states. `data/libero_lerobot` is the complete 34.9 GB demonstration dataset for offline diagnostics and future training; closed-loop evaluation does not require loading it into GPU memory.

## Core experimental design

Every comparison is paired by `(suite, task_id, init_seed)`. The frozen policy is always evaluated first and its trajectory is retained, then each adaptive condition is run from the identical initial state.

Conditions:

1. `frozen`: no update; estimates shift difficulty.
2. `online_persistent`: update after each eligible step and retain state across the episode.
3. `online_reset`: same update, reset adaptive state after a configured horizon.
4. `buffer_offline`: collect a rollout/buffer, adapt between episodes, never within the acting episode.

The runnable SmolVLA online conditions are explicitly a **self-supervised residual probe**, not a claimed reproduction of TTT-VLA or TT-VLA. At each update point, the frozen policy receives the current image and a weakly rotated view with identical flow noise. Their action disagreement is accumulated in a bounded 7-D residual. `online_persistent` retains that state, `online_reset` zeros it every 20 decisions, and `buffer_offline` estimates one fixed residual in a calibration rollout and applies it only on the replayed episode. The frozen model is never modified.

Expert demonstrations are used only retrospectively for labels. For each task, an 8-D state nearest-neighbour index predicts a reference action. The action-error threshold is the 90th percentile of leave-one-out expert prediction error; the state-support threshold is the 99th percentile of expert nearest-neighbour distance. An online update is labeled harmful only inside this support region, while entering an unsupported state counts as a closed-loop error. Neither signal is available to the acting policy or its update rule.

Shifts are swept independently before combinations are attempted:

- camera roll: `0, 2, 5, 10` degrees;
- object pose translation: `0, 1, 2, 4` cm in x/y, using deterministic initial-state edits;
- action translation bias: `0, 0.005, 0.01, 0.02` in normalized action units;
- gripper bias/delay: `0, 1, 2` control steps.

Minimum pilot: LIBERO-Spatial tasks 0–2, 10 seeds, all conditions. Confirm the metric pipeline and effect direction before the preregistered run. Main run: all 10 Spatial tasks, at least 30 paired seeds; repeat the strongest shift on Object and Goal.

## Primary metrics

- **Negative adaptation rate (NAR):** among paired episodes successful when frozen, the fraction that fail under adaptation.
- **Positive adaptation rate (PAR):** among frozen failures, the fraction rescued by adaptation. Report NAR and PAR together.
- **Error amplification:** risk ratio `P(E[t+k] | first harmful update at t) / P(E[t+k])` for horizons 1, 2, 4, 8. A task/seed-cluster bootstrap supplies confidence intervals.
- **Update necessity:** fraction of decisions where adaptation changes the executed action above a fixed norm threshold; split into beneficial, harmful, and neutral changes through paired replay.
- **Drift:** adaptive-state norm, action residual norm, base/adapted action disagreement, and success as functions of update count.
- **Online-buffer gap:** paired success and NAR difference between `online_persistent` and `buffer_offline` under an equal update budget.

An episode record uses the JSONL schema in `configs/record_schema.json`. Compute paired metrics with:

```bash
.venv/bin/python -m stability_ttt.metrics \
  --records results/episodes.jsonl \
  --output results/metrics.json
```

## Baseline smoke test

```bash
export MUJOCO_GL=egl
export HF_HOME="$PWD/.hf-cache"
source .venv/bin/activate

lerobot-eval \
  --policy.path="$PWD/checkpoints/smolvla_libero" \
  --policy.device=cuda \
  --policy.num_steps=10 \
  --policy.n_action_steps=1 \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  --output_dir=results/smoke
```

`n_action_steps=1` is intentional: per-step intervention and logging are impossible if a long cached action chunk bypasses the adaptation point.

## Experiment stages

```bash
# 3 tasks x 3 init states; all shift families and magnitudes
scripts/screen_shifts.sh
.venv/bin/python -m stability_ttt.select_shift \
  --records results/shift_screen.jsonl \
  --output results/shift_selection.json

# Supply the selected shift argument from shift_selection.json
scripts/run_pilot.sh --camera-roll-deg 10
scripts/analyze.sh results/pilot.jsonl

# Full Spatial paired run (10 tasks x 30 states x 4 conditions)
scripts/run_main_spatial.sh --camera-roll-deg 10
scripts/analyze.sh results/main_spatial.jsonl

# Object and Goal replication
scripts/run_replication.sh --camera-roll-deg 10
```

Pilot, main, and replication use a fixed batch size of three. Batch and sequential BF16 kernels can differ slightly, so results from the sequential screening are used only to select a shift and are never pooled with batched paired metrics. The batch runner still performs one-step feedback and maintains independent simulator, residual, noise stream, termination state, and log per seed.

All runners append one complete JSON record at a time and skip existing `(task, seed, condition, shift)` keys, so interrupted runs are restartable. Physically unstable object-pose edits are restored and recorded with `valid=false`; metrics exclude them rather than treating them as task failures. If an object-pose shift is selected, use the sequential runner because collision validation is environment-specific.

## Reproducibility boundaries

PDF is available as official code and is pinned under `third_party/`. TTT-VLA and TT-VLA are retained as paper-level comparison targets until runnable official code/checkpoints are released. Results from substitute implementations must be labeled `reimplementation`, never `official reproduction`.

## Completed diagnostic artifacts

The full run contains 2,000 valid paired records: 1,200 Spatial, 400 Object,
and 400 Goal. Compact metrics, plots, and the interpretation are in
[`results/diagnostic_report.md`](results/diagnostic_report.md). The additional
consolidation analysis is generated by `scripts/consolidation_analysis.py` and
is stored beside each suite's metrics. Raw JSONL rollout logs are tracked for
exact replay; downloaded model/data and third-party checkouts are excluded by
`.gitignore` and pinned in `configs/provenance.json`.
