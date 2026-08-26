# Oracle Selective Consolidation — 10-hour pilot

This is a time-bounded Gate 1 pilot, not the pre-registered 3-seed final
experiment.  It evaluates all 10 tasks in each LIBERO suite with the
predetermined seed 0, and performs exact terminal-success commit/rollback
counterfactuals for the first three adaptation decisions of each episode.
After the probe budget is exhausted, no unvalidated update is persisted.
The update generator, camera shift (10 degrees), and terminal-success oracle
are otherwise unchanged.

## Results

Success rates below are the oracle pilot (10 episodes) and the matched D0
seed-0 conditions.  NAR/PAR are paired against frozen on the same task/seed.

| Suite | Frozen SR | Persistent SR | Reset SR | Buffer SR | Oracle SR | Oracle NAR | Oracle PAR | Commit rate | CHR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spatial | 50% | 20% | 50% | 50% | 50% | 0/5 = 0% | 0/5 = 0% | 3/30 = 10.0% | 0/30 = 0% |
| Object | 30% | 40% | 30% | 40% | 40% | 0/3 = 0% | 1/7 = 14.3% | 3/30 = 10.0% | 0/30 = 0% |
| Goal | 70% | 70% | 80% | 80% | 70% | 1/7 = 14.3% | 1/3 = 33.3% | 4/30 = 13.3% | 0/30 = 0% |

Oracle utility counts (positive, zero, negative) are Spatial (3, 27, 0),
Object (3, 27, 0), and Goal (4, 26, 0).  The low 10–13% commit rate shows
that the oracle rejects most proposed persistent updates in this pilot, but
the pilot has too few frozen-failure episodes to establish rescue reliably.

Task-cluster bootstrap intervals are in the per-suite metric JSON files.  The
pilot estimates are necessarily discrete because there is one seed per task.
For reference, immutable D0 persistent-minus-reset NAR deltas were Spatial
15.4pp (95% CI 5.2–24.0pp), Object 49.0pp (34.6–66.7pp), and Goal 21.4pp
(8.7–38.1pp).

## Pre-registered decision

This pilot is **NO-GO for a deployable gate** under the stated success rule.
Although oracle NAR is lower than persistent in all three suites, the oracle
does not improve overall success by at least 5 percentage points over
persistent in two suites (Spatial +30pp, Object +0pp, Goal +0pp), and its
PAR is below 80% of reset PAR in Goal.  The result is therefore mechanism
evidence, not a claim that a practical gate is validated.  Do not train or
add a learned/deployable consolidation gate from this pilot alone.

## Reproducibility

```bash
MUJOCO_GL=egl HF_HOME="$PWD/.hf-cache" PYTHONPATH="$PWD" \
.venv/bin/python -m stability_ttt.runner \
  --checkpoint checkpoints/smolvla_libero --dataset-root data/libero_lerobot \
  --output results/oracle_gate10h3_libero_spatial.jsonl \
  --suite libero_spatial --task-ids 0,1,2,3,4,5,6,7,8,9 --seeds 0 \
  --conditions oracle_selective_commit --camera-roll-deg 10 \
  --oracle-max-updates 3
```

The Object and Goal runs use the same command with the suite and output path
changed.  Raw JSONL, metrics, and D0 baselines are committed alongside this
report.
