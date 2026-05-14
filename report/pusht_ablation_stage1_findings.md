# PushT Ablation Stage 1 Findings

This is a historical structural screen. It selected candidates for Stage 2 and
Stage 3; it is superseded by the Stage 3 test1000 result in
`report/pusht_stage3_protocol.md`.

Stage 1 was run locally on the RTX 4070 Laptop GPU from source commit `5a586929`.
Each row aggregates two evaluation seeds, `50` PushT episodes per seed.
AF means Appearance-Factored: the AF variants add an appearance branch and
appearance-related losses while the planner still uses the dynamics latent.

Pre-Stage-2 gate: after Stage 1, the eval goal synchronization path was hardened
and AF independence loss was changed from raw covariance to standardized
cross-covariance. Stage 2 then retrained the selected AF variants from scratch;
the Stage 1 table remains a historical selection signal for which structures
were scaled.

| Variant | Family | Seed 42 | Seed 43 | Successes | Episodes | Aggregate | Delta vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1_current` | v1 | 10.0 | 10.0 | 10 | 100 | 10.0 | +4.0 pp |
| `v2_app_nuisance_only` | v2 | 6.0 | 10.0 | 8 | 100 | 8.0 | +2.0 pp |
| `baseline` | LeWM | 4.0 | 8.0 | 6 | 100 | 6.0 | 0.0 pp |
| `v2_weak_grl` | v2 | 2.0 | 10.0 | 6 | 100 | 6.0 | 0.0 pp |
| `v2_grl_warmup` | v2 | 8.0 | 4.0 | 6 | 100 | 6.0 | 0.0 pp |
| `v1_indep_only` | v1 | 2.0 | 8.0 | 5 | 100 | 5.0 | -1.0 pp |
| `v1_seq_only` | v1 | 4.0 | 6.0 | 5 | 100 | 5.0 | -1.0 pp |
| `v2_current` | v2 | 2.0 | 8.0 | 5 | 100 | 5.0 | -1.0 pp |
| `v1_seq_stopgrad` | v1 | 2.0 | 4.0 | 3 | 100 | 3.0 | -3.0 pp |
| `v1_inv_only` | v1 | 2.0 | 2.0 | 2 | 100 | 2.0 | -4.0 pp |

Historical Stage 1 decision:

- Keep `v1_current` as the lead structure.
- Treat `v2_app_nuisance_only` as the only v2-family candidate worth scaling now.
- Do not scale full `v2_current`, `v2_weak_grl`, or `v2_grl_warmup` until a cleaner reason appears; they did not beat baseline under this budget.
- Stage 2 was scheduled to compare `baseline`, `v1_current`, and `v2_app_nuisance_only` across longer training and more eval seeds.

Current readout: Stage 3 later compared baseline and `v1_current` across five
train seeds with test1000 evaluation and found an effective tie, not a reliable
AF-LeWM v1 improvement.

Raw machine-readable outputs:

- `report/pusht_ablation_stage1_summary.csv`
- `report/pusht_ablation_stage1_summary.json`
