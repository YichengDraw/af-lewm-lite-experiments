# PushT Ablation Stage 2 Findings

Stage 2 was run locally on the RTX 4070 Laptop GPU. Training used source
commit `39ac7a30`; reports were regenerated with commit `1115bf8`. No rented
GPU was used.

Each variant was trained with two training seeds, `3072` and `3073`, for `50`
epochs. Each checkpoint was evaluated with four eval seeds, `42` through `45`,
and `50` PushT episodes per eval seed. That gives `400` eval episodes per
variant per checkpoint epoch.

## Aggregate Success Rate

| Checkpoint | Variant | Train seeds | Successes | Episodes | Success rate | Delta vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| epoch 25 | `baseline` | 3072, 3073 | 21 | 400 | 5.25% | 0.00 pp |
| epoch 25 | `v1_current` | 3072, 3073 | 19 | 400 | 4.75% | -0.50 pp |
| epoch 25 | `v2_app_nuisance_only` | 3072, 3073 | 20 | 400 | 5.00% | -0.25 pp |
| epoch 50 | `baseline` | 3072, 3073 | 25 | 400 | 6.25% | 0.00 pp |
| epoch 50 | `v1_current` | 3072, 3073 | 22 | 400 | 5.50% | -0.75 pp |
| epoch 50 | `v2_app_nuisance_only` | 3072, 3073 | 20 | 400 | 5.00% | -1.25 pp |

Per-training-seed detail:

| Checkpoint | Variant | Train seed | Successes | Episodes | Success rate |
| --- | --- | ---: | ---: | ---: | ---: |
| epoch 25 | `baseline` | 3072 | 11 | 200 | 5.50% |
| epoch 25 | `baseline` | 3073 | 10 | 200 | 5.00% |
| epoch 25 | `v1_current` | 3072 | 10 | 200 | 5.00% |
| epoch 25 | `v1_current` | 3073 | 9 | 200 | 4.50% |
| epoch 25 | `v2_app_nuisance_only` | 3072 | 8 | 200 | 4.00% |
| epoch 25 | `v2_app_nuisance_only` | 3073 | 12 | 200 | 6.00% |
| epoch 50 | `baseline` | 3072 | 14 | 200 | 7.00% |
| epoch 50 | `baseline` | 3073 | 11 | 200 | 5.50% |
| epoch 50 | `v1_current` | 3072 | 10 | 200 | 5.00% |
| epoch 50 | `v1_current` | 3073 | 12 | 200 | 6.00% |
| epoch 50 | `v2_app_nuisance_only` | 3072 | 13 | 200 | 6.50% |
| epoch 50 | `v2_app_nuisance_only` | 3073 | 7 | 200 | 3.50% |

Stage 1 screened every planned v1/v2 variant with one training seed and `100`
episodes per structure. Stage 2 then scaled the leading v1-family and v2-family
candidates only.

| Stage 1 variant | Family | Successes | Episodes | Success rate | Delta vs baseline | Stage 2 status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `baseline` | LeWM | 6 | 100 | 6.0 | 0.0 pp | scaled |
| `v1_current` | v1 | 10 | 100 | 10.0 | +4.0 pp | scaled |
| `v1_inv_only` | v1 | 2 | 100 | 2.0 | -4.0 pp | stopped |
| `v1_indep_only` | v1 | 5 | 100 | 5.0 | -1.0 pp | stopped |
| `v1_seq_only` | v1 | 5 | 100 | 5.0 | -1.0 pp | stopped |
| `v1_seq_stopgrad` | v1 | 3 | 100 | 3.0 | -3.0 pp | stopped |
| `v2_app_nuisance_only` | v2 | 8 | 100 | 8.0 | +2.0 pp | scaled |
| `v2_weak_grl` | v2 | 6 | 100 | 6.0 | 0.0 pp | stopped |
| `v2_current` | v2 | 5 | 100 | 5.0 | -1.0 pp | stopped |
| `v2_grl_warmup` | v2 | 6 | 100 | 6.0 | 0.0 pp | stopped |

## Readout

The exact answer to the baseline comparison is stage-dependent. In Stage 1,
`v1_current` and `v2_app_nuisance_only` were above baseline, and three other
v2-family rows tied or trailed baseline. In Stage 2, both scaled AF candidates
were below baseline at epoch 25 and epoch 50. The margins are small in absolute
success count, but the larger reliability check does not support AF-LeWM-lite as
a robust PushT improvement under this setup.

`v1_current` has the cleaner factorization diagnostics: epoch 50
`diag_emb_app_cross_cov` is about `0.035-0.036`, and app augmentation
sensitivity stays near zero. That means the v1 factorization objective is doing
something measurable, but the cleaner latent split does not translate into
better MPC success here.

`v2_app_nuisance_only` learns stronger appearance/nuisance information in
`app_emb`, but its dynamics/app cross-cov remains much higher: about
`0.36-0.40` at epoch 50. Its success is also less stable across train seeds
(`6.5%` vs `3.5%`). This supports the simpler interpretation that the v2
nuisance branch adds optimization pressure without reliably improving the
planning latent.

## Decision

Use baseline as the main reliable PushT result for this repo. Keep v1 as an
interesting negative/diagnostic result: it creates a cleaner appearance split,
but current planning success does not improve. Do not invest further in the
current v2 family without simplifying the objective or changing the training
scale/eval protocol.

Raw machine-readable outputs:

- `report/pusht_ablation_stage2_epoch25_summary.csv`
- `report/pusht_ablation_stage2_epoch25_summary.json`
- `report/pusht_ablation_stage2_epoch50_summary.csv`
- `report/pusht_ablation_stage2_epoch50_summary.json`
