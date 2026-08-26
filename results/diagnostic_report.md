# Closed-Loop VLA Test-Time Learning Stability

## Executive result

The experiment isolates a sharper failure mode than “online TTT is unstable”:

> Local adaptation can be useful; unconditional persistent consolidation is harmful.

The same per-step update is relatively safe when reset, but harmful when every
update is written into a persistent adaptive state. This is the adaptation vs.
consolidation gap.

## Setup

- Model: official `HuggingFaceVLA/smolvla_libero` (605M), run on an RTX 3080
  10GB with BF16 and batch size 3.
- Data: complete LIBERO-LeRobot export (1,693 episodes, 273,465 frames,
  approximately 33GB).
- Shift: screened strongest reproducible shift, `camera roll = 10°`;
  `shift_id=cam10_obj0_0_act0_grip0`.
- Conditions: `frozen`, `online_persistent`, `online_reset`, and
  `buffer_offline`.
- Pairing: identical task/initial-state/seed across conditions. Reference
  expert traces are retrospective diagnostics only and never drive an update.

## Completed runs

| suite | tasks | seeds/task | conditions | records |
|---|---:|---:|---:|---:|
| LIBERO-Spatial | 10 | 30 | 4 | 1,200 |
| LIBERO-Object | 10 | 10 | 4 | 400 |
| LIBERO-Goal | 10 | 10 | 4 | 400 |

All 2,000 records are valid, non-empty, batch size 3, and use the selected
shift. Every record contains per-step logs.

## Paired corruption and rescue

Corruption is `frozen success -> adapted failure` (NAR numerator). Rescue is
`frozen failure -> adapted success` (PAR numerator).

| suite / condition | corruption | NAR | rescue | PAR |
|---|---:|---:|---:|---:|
| Spatial / persistent | 64/162 | 39.5% | 35/138 | 25.4% |
| Spatial / reset | 39/162 | 24.1% | 25/138 | 18.1% |
| Spatial / buffer | 21/162 | 13.0% | 24/138 | 17.4% |
| Object / persistent | 30/49 | 61.2% | 13/51 | 25.5% |
| Object / reset | 6/49 | 12.2% | 10/51 | 19.6% |
| Object / buffer | 2/49 | 4.1% | 5/51 | 9.8% |
| Goal / persistent | 24/70 | 34.3% | 9/30 | 30.0% |
| Goal / reset | 9/70 | 12.9% | 7/30 | 23.3% |
| Goal / buffer | 4/70 | 5.7% | 7/30 | 23.3% |

The descriptive Fisher test on frozen-success episodes gives persistent vs.
buffer corruption p-values of `6.96e-8` (Spatial), `6.91e-10` (Object), and
`3.22e-5` (Goal). Persistent vs. reset is also significant in Object
(`7.13e-7`) and Goal (`4.83e-3`). Paired bootstrap intervals are the primary
uncertainty summaries in the JSON metrics.

## Interpretation

Persistent adaptation changes almost every decision (Spatial 98.9%, Object
99.0%, Goal 99.3%), while reset and buffer conditions are less invasive. The
persistent adaptive-state norm and action deviation increase with cumulative
update count. This supports:

`noisy local update -> persistent consolidation -> adaptive drift -> base-policy corruption`.

The result does **not** establish that every harmful update recursively raises
the next-step error probability: the current propagation risk ratios are below
one at the tested horizons. The robust claim is about accumulation and final
episode corruption, not a stronger universal collapse theorem.

## Next method implied by the diagnosis

Do not add a new representation first. Separate:

`propose -> act -> validate -> consolidate`.

An update is speculative until environmental evidence supports committing it:

`c_tilde = c_t + Delta_t`, then `c_(t+1) = c_t + gamma_t Delta_t`,

where `gamma_t` is an evidence-based commit/rollback decision. This is the
minimal intervention between the observed reset and persistent baselines.

## Reproduction artifacts

- [Spatial records](main_spatial.jsonl)
- [Spatial metrics](main_spatial_metrics.json)
- [Spatial consolidation analysis](main_spatial_consolidation.json)
- [Object records](replication_libero_object.jsonl)
- [Object metrics](replication_libero_object_metrics.json)
- [Object consolidation analysis](replication_libero_object_consolidation.json)
- [Goal records](replication_libero_goal.jsonl)
- [Goal metrics](replication_libero_goal_metrics.json)
- [Goal consolidation analysis](replication_libero_goal_consolidation.json)
- [Plots](main_spatial_consolidation.png), [Object plots](replication_libero_object_consolidation.png), [Goal plots](replication_libero_goal_consolidation.png)

The implementation and exact commands are documented in the repository
README and `scripts/`.
