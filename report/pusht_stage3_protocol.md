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

Training saves object checkpoints every 5 epochs. Closed-loop PushT validation also runs every 5 epochs. This is denser than the previous 25/50-only schedule because the current question is training stability, not only final success.

Final test evaluation is run once per model/seed using the checkpoint selected by validation closed-loop success. Validation loss is diagnostic; closed-loop success is the selection metric.

## Existing Remote Baseline

The SSH 5090 host already contains a PushT LeWM checkpoint trained with batch size 128 and learning rate 5e-5 for 10 epochs. It is useful as a reference row and smoke sanity check. It is not the primary Stage 3 baseline because it lacks the full Stage 3 split, manifest, W&B, and 100-epoch provenance.
