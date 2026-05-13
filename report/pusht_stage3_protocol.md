# PushT Stage 3 Protocol

Stage 3 tests whether AF-LeWM v1 improves over the official LeWM baseline on PushT under matched training and evaluation conditions.

## Training Anchor

The baseline hyperparameters follow the upstream LeWM PushT config:

- 100 epochs
- batch size 128
- 6 dataloader workers with persistent workers and prefetch factor 3
- AdamW, learning rate 5e-5, weight decay 1e-3
- bf16 precision
- SIGReg weight 0.09
- W&B enabled for serious runs

AF-LeWM v1 keeps the same settings and adds only the appearance-factored branch:

- appearance latent dimension 64
- appearance invariance weight 0.1
- dynamics/appearance cross-covariance independence weight 0.05
- the existing color/noise appearance augmentation policy

These AF-specific values define the v1 architecture for the primary comparison. They are not tuned during Stage 3.

On the RTX 5090 32 GB host, AF-LeWM v1 does not fit batch size 128 because it performs the clean, aug_a, and aug_b encoder passes in the same training step. If the primary run uses the 32 GB host, both baseline and v1 are run with the same fallback batch size 96. The report must mark that run as official-aligned except for the matched batch-size fallback.

A full official 100-epoch pass over all 1.58M train clips is not executable as a full 5-seed baseline/v1 study on a single RTX 5090 in a reasonable wall-clock budget. The executable 5090 profile is therefore:

- run label: `b96k1000e50`
- train seeds: `3072` through `3076`
- variants: baseline and v1
- max epoch: `50`
- train batches per epoch: `1000`
- validation batches per epoch: `50`
- closed-loop validation: every `5` epochs on the locked val100 manifest

This profile keeps the official model, optimizer, learning rate, precision, data split, and planner eval, while making the training budget explicit and finishable. It is a budgeted stability comparison, not a claim that the official full-data 100-epoch LeWM result has been reproduced.

The default W&B destination is entity `yicheng132024-southern-university-of-science-technology`, project `af-lewm-lite-stage3`. These can be overridden with `WANDB_ENTITY` and `WANDB_PROJECT`.

## Splits And Manifests

All runs use the same episode-disjoint split:

- train/val/test = 0.8/0.1/0.1
- split seed = 9001
- normalizers are fit only on train episodes

Evaluation uses locked manifests generated before evaluation:

- small validation: 100 starts
- large validation: 500 starts, available for heavier checks
- final test: 1000 starts

Every model and checkpoint evaluated against a manifest must use exactly the same row indices, episodes, and start steps. Paired reporting fails if manifests differ.

## Checkpoint And Evaluation Schedule

Training saves object checkpoints every 5 epochs. The runner trains in resumable 5-epoch chunks, runs closed-loop PushT validation after each chunk, then resumes from the Lightning weights checkpoint. This is denser than the previous 25/50-only schedule because the current question is training stability, not only final success.

Final test evaluation is run once per model/seed using the checkpoint selected by validation closed-loop success. Validation loss is diagnostic; closed-loop success is the selection metric.

## Completed `b96k1000e50` Results

This run completed on the RTX 5090 host with 5 train seeds, val100 checkpoint selection, and final test1000 PushT success-rate evaluation. The generated machine-readable artifacts are:

- `report/pusht_stage3_v1_b96k1000e50_summary.csv`
- `report/pusht_stage3_v1_b96k1000e50_paired.csv`
- `report/pusht_stage3_v1_b96k1000e50_val_curve.csv`
- JSON companions for the same tables
- locked manifests under `report/stage3_manifests/`

Per-seed results:

| train seed | baseline best val100 | baseline test1000 | v1 best val100 | v1 test1000 | v1 - baseline test |
|---:|---:|---:|---:|---:|---:|
| 3072 | 90.0 | 87.1 | 88.0 | 87.1 | 0.0 |
| 3073 | 82.0 | 80.5 | 81.0 | 78.9 | -1.6 |
| 3074 | 81.0 | 81.6 | 81.0 | 84.5 | +2.9 |
| 3075 | 85.0 | 84.6 | 84.0 | 84.5 | -0.1 |
| 3076 | 88.0 | 87.2 | 85.0 | 86.1 | -1.1 |

Aggregate results:

| variant | best val100 mean | best val100 std | test1000 mean | test1000 std |
|---|---:|---:|---:|---:|
| baseline | 85.20 | 3.83 | 84.20 | 3.08 |
| AF-LeWM v1 | 83.80 | 2.95 | 84.22 | 3.17 |

The paired mean test delta is `+0.02` percentage points for v1 over baseline with standard deviation `1.75` points across the 5 paired seeds. The evidence does not show a reliable v1 advantage under this budgeted protocol. It also does not show a meaningful baseline advantage on final test1000: the mean test difference is effectively zero, while the seed-to-seed spread is several percentage points.

## Existing Remote Baseline

The SSH 5090 host already contains a PushT LeWM checkpoint trained with batch size 128 and learning rate 5e-5 for 10 epochs. It is useful as a reference row and smoke sanity check. It is not the primary Stage 3 baseline because it lacks the full Stage 3 split, manifest, W&B, and 100-epoch provenance.
