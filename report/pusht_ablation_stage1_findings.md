# PushT Ablation Stage 1 Findings

Stage 1 was run locally on the RTX 4070 Laptop GPU from source commit `5a586929`.
Each row aggregates two evaluation seeds, `50` PushT episodes per seed.

| Variant | Seed 42 | Seed 43 | Aggregate |
| --- | ---: | ---: | ---: |
| `v1_current` | 10.0 | 10.0 | 10.0 |
| `v2_app_nuisance_only` | 6.0 | 10.0 | 8.0 |
| `baseline` | 4.0 | 8.0 | 6.0 |
| `v2_weak_grl` | 2.0 | 10.0 | 6.0 |
| `v2_grl_warmup` | 8.0 | 4.0 | 6.0 |
| `v1_indep_only` | 2.0 | 8.0 | 5.0 |
| `v1_seq_only` | 4.0 | 6.0 | 5.0 |
| `v2_current` | 2.0 | 8.0 | 5.0 |
| `v1_seq_stopgrad` | 2.0 | 4.0 | 3.0 |
| `v1_inv_only` | 2.0 | 2.0 | 2.0 |

Current decision:

- Keep `v1_current` as the lead structure.
- Treat `v2_app_nuisance_only` as the only v2-family candidate worth scaling now.
- Do not scale full `v2_current`, `v2_weak_grl`, or `v2_grl_warmup` until a cleaner reason appears; they did not beat baseline under this budget.
- Stage 2 should compare `baseline`, `v1_current`, and `v2_app_nuisance_only` across longer training and more eval seeds.

Raw machine-readable outputs:

- `report/pusht_ablation_stage1_summary.csv`
- `report/pusht_ablation_stage1_summary.json`
