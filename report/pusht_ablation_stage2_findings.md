# PushT Ablation Stage 2 Findings

Stage 2 was run locally on the RTX 4070 Laptop GPU. Training used source
commit `39ac7a30`; reports were regenerated with commit `1115bf8`. No rented
GPU was used.

Each variant was trained with two training seeds, `3072` and `3073`, for `50`
epochs. Each checkpoint was evaluated with four eval seeds, `42` through `45`,
and `50` PushT episodes per eval seed. That gives `400` eval episodes per
variant per checkpoint epoch.

## Aggregate Success Rate

| Checkpoint | Variant | Successes | Episodes | Success rate |
| --- | --- | ---: | ---: | ---: |
| epoch 25 | `baseline` | 21 | 400 | 5.25 |
| epoch 25 | `v1_current` | 19 | 400 | 4.75 |
| epoch 25 | `v2_app_nuisance_only` | 20 | 400 | 5.00 |
| epoch 50 | `baseline` | 25 | 400 | 6.25 |
| epoch 50 | `v1_current` | 22 | 400 | 5.50 |
| epoch 50 | `v2_app_nuisance_only` | 20 | 400 | 5.00 |

## Readout

The Stage 1 v1 lead did not survive the longer two-seed Stage 2 check. At both
epoch 25 and epoch 50, baseline is the strongest aggregate result. The margins
are small in absolute success count, but the direction is consistent enough that
AF-LeWM-lite should not be claimed as a robust PushT improvement under this
setup.

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
