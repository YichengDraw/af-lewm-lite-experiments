# PushT AF-LeWM-lite Ablation Protocol

## Goal

Select the structure worth further investment: baseline LeWM, v1, a simplified v2, or full v2.
The decision uses planning success as the hard metric and latent diagnostics as mechanism evidence.

## Stage 1: Structural Screen

Run the current short reliable budget for all Stage 1 variants:

```powershell
python tools/run_pusht_ablation.py --mode all --stage stage1
```

The runner reuses the existing reliable `baseline`, `v1_current`, and `v2_current` artifacts when present. New variants isolate the contribution of invariance, cross-cov independence, sequence-consistent augmentation, stop-gradient target, appearance nuisance prediction, weak GRL, and GRL warmup.

Stage 1 uses:

- Train seed: `3072`
- Train budget: `10` epochs, `200` train batches per epoch, batch size `4`
- Eval seeds: `42`, `43`
- Eval episodes: `50` per seed, `100` per structure

## Stage 2: Scale Test

After Stage 1, choose baseline plus the best v1-family and v2-family variants. Run:

```powershell
python tools/run_pusht_ablation.py --stage stage2 --mode all --ids baseline v1_current v2_app_nuisance_only
```

Stage 2 uses:

- Train seeds: `3072`, `3073`
- Train budget: `50` epochs
- Object checkpoints every `5` epochs
- Eval seeds: `42`, `43`, `44`, `45`
- Default eval epoch: `50`

To inspect intermediate checkpoints:

```powershell
python tools/run_pusht_ablation.py --stage stage2 --mode eval --ids baseline v1_current v2_app_nuisance_only --eval-epoch 25
python tools/run_pusht_ablation.py --stage stage2 --mode report --ids baseline v1_current v2_app_nuisance_only --eval-epoch 25
```

## Decision Rule

Prefer the simplest structure that wins or ties within noise.

- Promote v1 if it stays ahead of v2-family in Stage 1 and Stage 2.
- Promote a simplified v2 only if it beats v1 on aggregate success and shows better latent factorization.
- Treat full v2 as exploratory unless it beats v1 consistently across both Stage 2 train seeds.
- If baseline catches the AF variants, pause the AF claim and inspect augmentation/loss mismatch before scaling further.

## Outputs

The runner writes:

- `report/pusht_ablation_stage1_summary.csv`
- `report/pusht_ablation_stage1_summary.json`
- `report/pusht_ablation_stage2_epoch25_summary.csv` for intermediate checks
- `report/pusht_ablation_stage2_epoch25_summary.json` for intermediate checks
- `report/pusht_ablation_stage2_epoch50_summary.csv`
- `report/pusht_ablation_stage2_epoch50_summary.json`

Each run directory also receives `latent_diagnostics.json`.
