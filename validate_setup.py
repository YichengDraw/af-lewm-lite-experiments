"""Validate the le-wm reproduction setup."""
import os
import sys
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

if os.name == "nt":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

errors = []
warnings = []
PUSHT_MIN_OFFICIAL_BYTES = 1_000_000_000

print("=" * 60)
print("LeWM Reproduction Setup Validation")
print("=" * 60)

# 1. Core imports
print("\n[1/7] Checking core imports...")
try:
    import torch
    import stable_worldmodel as swm
    import stable_pretraining as spt
    import lightning as pl
    import hydra
    import einops
    import h5py
    import sklearn
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP, SIGReg
    from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
    print("  OK - All imports successful")
except ImportError as e:
    errors.append(f"Import error: {e}")
    print(f"  FAIL - {e}")

# 2. PyTorch CUDA
print("\n[2/7] Checking PyTorch CUDA...")
print(f"  PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    # Check bf16 support
    if torch.cuda.get_device_capability()[0] >= 8:
        print("  BF16 supported: Yes")
    else:
        warnings.append("GPU does not support BF16 - may need to change precision config")
        print("  BF16 supported: No (warning)")
else:
    errors.append("CUDA not available - GPU training/eval will not work")
    print("  FAIL - CUDA not available")

# 3. Dataset
print("\n[3/7] Checking dataset...")
cache_dir = Path(swm.data.utils.get_cache_dir())
print(f"  STABLEWM_HOME: {cache_dir}")
pusht_path = cache_dir / "pusht_expert_train.h5"
if pusht_path.exists():
    ds = swm.data.HDF5Dataset("pusht_expert_train", keys_to_cache=["action", "proprio", "state"])
    print(f"  pusht_expert_train.h5: OK ({len(ds)} samples)")
    if pusht_path.stat().st_size < PUSHT_MIN_OFFICIAL_BYTES:
        warnings.append(
            "pusht_expert_train.h5 is a small placeholder/debug file, not the official paper dataset"
        )
        print("  Warning: current PushT file is a placeholder/debug dataset")
else:
    errors.append(f"Dataset not found: {pusht_path}")
    print(f"  FAIL - {pusht_path} not found")

# Check other datasets
for name in ["tworoom", "dmc/reacher_random", "ogbench/cube_single_expert"]:
    p = cache_dir / f"{name}.h5"
    if p.exists():
        print(f"  {name}.h5: OK")
    else:
        warnings.append(f"Dataset not found: {name}.h5 (needed for non-PushT experiments)")
        print(f"  {name}.h5: NOT FOUND (optional)")

# 4. Checkpoints
print("\n[4/7] Checking checkpoints...")
runs_dir = cache_dir / "runs" / "pusht_expert_train" / "lewm"
if runs_dir.exists():
    ckpts = list(runs_dir.glob("*_object.ckpt"))
    if ckpts:
        ckpts.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        print(f"  Trained checkpoints: {len(ckpts)} found")
        print(f"  Latest: {ckpts[0].name}")
    else:
        warnings.append("No checkpoints found - need to train first")
        print("  No checkpoints found")
else:
    warnings.append(f"Runs directory not found: {runs_dir}")
    print(f"  Runs directory not found: {runs_dir}")

# 5. Model instantiation
print("\n[5/7] Testing model instantiation...")
try:
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = 192
    predictor = ARPredictor(
        num_frames=3, input_dim=embed_dim, hidden_dim=hidden_dim,
        output_dim=hidden_dim, depth=6, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.1, emb_dropout=0.0
    )
    action_encoder = Embedder(input_dim=10, emb_dim=embed_dim)  # pusht action_dim*frameskip = 2*5=10
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    world_model = JEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=projector, pred_proj=predictor_proj
    )
    total_params = sum(p.numel() for p in world_model.parameters())
    print(f"  OK - Model created with {total_params/1e6:.1f}M parameters")
except Exception as e:
    errors.append(f"Model instantiation failed: {e}")
    print(f"  FAIL - {e}")

# 6. Checkpoint loading
print("\n[6/7] Testing checkpoint loading...")
if runs_dir.exists() and ckpts:
    try:
        model = swm.policy.AutoCostModel("runs/pusht_expert_train/lewm")
        print(f"  OK - Checkpoint loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        has_cost = hasattr(model, "get_cost")
        print(f"  Has get_cost: {has_cost}")
    except Exception as e:
        errors.append(f"Checkpoint loading failed: {e}")
        print(f"  FAIL - {e}")
else:
    print("  SKIP - No checkpoints to load")

# 7. Environment
print("\n[7/7] Testing environment creation...")
try:
    world = swm.World(env_name="swm/PushT-v1", num_envs=1, max_episode_steps=100, image_shape=(224, 224))
    print(f"  OK - PushT environment created")
    world.close()
except Exception as e:
    errors.append(f"Environment creation failed: {e}")
    print(f"  FAIL - {e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if errors:
    print(f"\nERRORS ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
if warnings:
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  - {w}")
if not errors:
    if any("placeholder/debug" in w for w in warnings):
        print("\nCORE SETUP IS READY. OFFICIAL FULL-DATA REPRODUCTION IS STILL IN PROGRESS.")
    else:
        print("\nSETUP IS READY FOR REPRODUCTION!")
    print("\nCommands to run:")
    print("  Training:   python train.py data=pusht")
    print("  Evaluation: python eval.py --config-name=pusht.yaml policy=runs/pusht_expert_train/lewm")
else:
    print(f"\n{len(errors)} error(s) must be fixed before running.")
