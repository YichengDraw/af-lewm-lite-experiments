"""Validate the AF-LeWM PushT Stage 3 setup."""
import csv
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
ROOT = Path(__file__).resolve().parent
STAGE3_SUMMARY = ROOT / "report" / "pusht_stage3_v1_b96k1000e50_summary.csv"
STAGE3_REQUIRED_ARTIFACTS = [
    ROOT / "report" / "pusht_stage3_protocol.md",
    ROOT / "report" / "pusht_stage3_v1_b96k1000e50_summary.csv",
    ROOT / "report" / "pusht_stage3_v1_b96k1000e50_paired.csv",
    ROOT / "report" / "pusht_stage3_v1_b96k1000e50_val_curve.csv",
    ROOT / "report" / "stage3_manifests" / "pusht_stage3_val_n100_seed9100.json",
    ROOT / "report" / "stage3_manifests" / "pusht_stage3_test_n1000_seed9200.json",
]


print("=" * 60)
print("AF-LeWM PushT Setup Validation")
print("=" * 60)

print("\n[1/6] Checking core imports...")
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
except ImportError as exc:
    print(f"  FAIL - {exc}")
    sys.exit(1)

print("\n[2/6] Checking PyTorch CUDA...")
print(f"  PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    if torch.cuda.get_device_capability()[0] >= 8:
        print("  BF16 supported: Yes")
    else:
        warnings.append("GPU does not support BF16; change precision before training.")
        print("  BF16 supported: No")
else:
    errors.append("CUDA not available; GPU training/eval will not work.")
    print("  FAIL - CUDA not available")

print("\n[3/6] Checking official PushT dataset...")
cache_dir = Path(swm.data.utils.get_cache_dir())
print(f"  STABLEWM_HOME: {cache_dir}")
pusht_path = cache_dir / "pusht_expert_train.h5"
if pusht_path.exists():
    ds = swm.data.HDF5Dataset("pusht_expert_train", keys_to_cache=["action", "proprio", "state"])
    print(f"  pusht_expert_train.h5: OK ({len(ds)} samples)")
    if pusht_path.stat().st_size < PUSHT_MIN_OFFICIAL_BYTES:
        errors.append("pusht_expert_train.h5 is smaller than the official dataset threshold.")
        print("  FAIL - current PushT file is too small for the final study.")
else:
    errors.append(f"Dataset not found: {pusht_path}")
    print(f"  FAIL - {pusht_path} not found")

print("\n[4/6] Checking Stage 3 repository artifacts...")
for artifact in STAGE3_REQUIRED_ARTIFACTS:
    if artifact.exists():
        print(f"  OK - {artifact.relative_to(ROOT)}")
    else:
        errors.append(f"Missing required Stage 3 artifact: {artifact.relative_to(ROOT)}")
        print(f"  FAIL - {artifact.relative_to(ROOT)}")

stage3_policies = []
if STAGE3_SUMMARY.exists():
    with STAGE3_SUMMARY.open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["output_model_name"]
            epoch = int(float(row["best_val_epoch"]))
            stage3_policies.append(
                f"runs/pusht_expert_train/{name}/{name}_epoch_{epoch}"
            )

print("\n[5/6] Testing model instantiation...")
try:
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = 192
    predictor = ARPredictor(
        num_frames=3,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=10, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )
    total_params = sum(p.numel() for p in world_model.parameters())
    print(f"  OK - Model created with {total_params / 1e6:.1f}M parameters")
except Exception as exc:
    errors.append(f"Model instantiation failed: {exc}")
    print(f"  FAIL - {exc}")

print("\n[6/6] Testing environment and checkpoint loading...")
try:
    world = swm.World(env_name="swm/PushT-v1", num_envs=1, max_episode_steps=100, image_shape=(224, 224))
    print("  OK - PushT environment created")
    world.close()
except Exception as exc:
    errors.append(f"Environment creation failed: {exc}")
    print(f"  FAIL - {exc}")

found_policies = []
for policy in stage3_policies:
    ckpt = cache_dir / f"{policy}_object.ckpt"
    if ckpt.exists():
        found_policies.append(policy)

if found_policies:
    for policy in found_policies:
        try:
            model = swm.policy.AutoCostModel(policy)
            has_get_cost = hasattr(model, "get_cost")
            if not has_get_cost:
                raise AttributeError("loaded object does not expose get_cost")
            print(f"  OK - Checkpoint loaded: {policy}")
            print(f"       Has get_cost: {has_get_cost}")
        except Exception as exc:
            errors.append(f"Checkpoint loading failed for {policy}: {exc}")
            print(f"  FAIL - {policy}: {exc}")
else:
    print("  SKIP - No local Stage 3 checkpoints found; report artifacts are still checked.")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if errors:
    print(f"\nERRORS ({len(errors)}):")
    for item in errors:
        print(f"  - {item}")
if warnings:
    print(f"\nWARNINGS ({len(warnings)}):")
    for item in warnings:
        print(f"  - {item}")

if errors:
    print(f"\n{len(errors)} error(s) must be fixed before running.")
    sys.exit(1)

print("\nSETUP IS READY FOR THE STAGE 3 PUSHT CODEPATH.")
print("\nCommands:")
print("  Status: python tools/run_pusht_stage3.py --mode status")
print("  Run:    python tools/run_pusht_stage3.py --mode cycle")
