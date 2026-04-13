"""
Reproduction helper for baseline LeWM and AF-LeWM-lite experiments.

Usage:
    python run_all.py --mode train    # Train selected environments
    python run_all.py --mode eval     # Evaluate selected environments
    python run_all.py --mode both     # Train then evaluate
    python run_all.py --mode status   # Check datasets and checkpoints
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

if os.name == "nt":
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

CACHE_DIR = Path(os.environ["STABLEWM_HOME"])
PUSHT_MIN_OFFICIAL_BYTES = 1_000_000_000

# All environments and their configs
ENVS = {
    "pusht": {
        "train_data": "pusht",
        "train_dataset_file": "pusht_expert_train.h5",
        "eval_config": "pusht.yaml",
        "eval_dataset_file": "pusht_expert_train.h5",
        "eval_policy_trained": "runs/pusht_expert_train/lewm",
        "eval_policy_pretrained": "pusht/lewm",
    },
    "tworoom": {
        "train_data": "tworoom",
        "train_dataset_file": "tworoom.h5",
        "eval_config": "tworoom.yaml",
        "eval_dataset_file": "tworoom.h5",
        "eval_policy_trained": "runs/tworoom/lewm",
        "eval_policy_pretrained": "tworoom/lewm",
    },
    "reacher": {
        "train_data": "dmc",
        "train_dataset_file": "reacher.h5",
        "eval_config": "reacher.yaml",
        "eval_dataset_file": "dmc/reacher_random.h5",
        "eval_policy_trained": "runs/reacher/lewm",
        "eval_policy_pretrained": "reacher/lewm",
    },
    "cube": {
        "train_data": "ogb",
        "train_dataset_file": "ogbench/cube_single_expert.h5",
        "eval_config": "cube.yaml",
        "eval_dataset_file": "ogbench/cube_single_expert.h5",
        "eval_policy_trained": "runs/ogbench/cube_single_expert/lewm",
        "eval_policy_pretrained": "cube/lewm",
    },
}


def check_status():
    """Check what datasets and checkpoints are available."""
    print("=" * 70)
    print("AF-LeWM-lite Experiment Status")
    print(f"STABLEWM_HOME: {CACHE_DIR}")
    print("=" * 70)

    for env_name, cfg in ENVS.items():
        print(f"\n--- {env_name.upper()} ---")

        # Check training dataset
        train_ds = CACHE_DIR / cfg["train_dataset_file"]
        ds_status = "OK" if train_ds.exists() else "MISSING"
        ds_size = f"({train_ds.stat().st_size / 1e9:.2f} GB)" if train_ds.exists() else ""
        if env_name == "pusht" and train_ds.exists() and train_ds.stat().st_size < PUSHT_MIN_OFFICIAL_BYTES:
            ds_status = "PLACEHOLDER"
        print(f"  Train dataset:  [{ds_status}] {cfg['train_dataset_file']} {ds_size}")

        # Check eval dataset (might differ from train)
        eval_ds = CACHE_DIR / cfg["eval_dataset_file"]
        if cfg["eval_dataset_file"] != cfg["train_dataset_file"]:
            ds_status = "OK" if eval_ds.exists() else "MISSING"
            ds_size = f"({eval_ds.stat().st_size / 1e9:.2f} GB)" if eval_ds.exists() else ""
            print(f"  Eval dataset:   [{ds_status}] {cfg['eval_dataset_file']} {ds_size}")

        # Check trained checkpoint
        trained_dir = CACHE_DIR / cfg["eval_policy_trained"]
        if trained_dir.exists():
            ckpts = list(trained_dir.glob("*_object.ckpt"))
            print(f"  Trained ckpt:   [OK] {len(ckpts)} checkpoints")
        else:
            print(f"  Trained ckpt:   [NONE] (train first)")

        # Check pretrained checkpoint
        pretrained_dir = CACHE_DIR / cfg["eval_policy_pretrained"]
        if pretrained_dir.exists() and list(pretrained_dir.glob("*_object.ckpt")):
            print(f"  Pretrained ckpt:[OK]")
        else:
            print(f"  Pretrained ckpt:[MISSING] (run download_and_convert_checkpoints.py)")

    # Check for archives that need extraction
    archives = list(CACHE_DIR.glob("*.tar.zst")) + list(CACHE_DIR.glob("*.h5.zst"))
    if archives:
        print(f"\n--- PENDING EXTRACTION ---")
        for a in archives:
            print(f"  {a.name} ({a.stat().st_size / 1e9:.2f} GB) -> run extract_datasets.py")


def run_train(env_name: str, extra_args: list = None):
    """Train LeWM for a specific environment."""
    cfg = ENVS[env_name]

    # Check dataset exists
    ds_path = CACHE_DIR / cfg["train_dataset_file"]
    if not ds_path.exists():
        print(f"  SKIP {env_name}: dataset {cfg['train_dataset_file']} not found")
        return False

    cmd = [
        sys.executable, "train.py",
        f"data={cfg['train_data']}",
        "wandb.enabled=False",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Training {env_name}: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_eval(env_name: str, use_pretrained: bool = True, extra_args: list = None):
    """Evaluate LeWM for a specific environment."""
    cfg = ENVS[env_name]

    policy_key = "eval_policy_pretrained" if use_pretrained else "eval_policy_trained"
    policy = cfg[policy_key]

    # Check checkpoint exists
    policy_dir = CACHE_DIR / policy
    if not policy_dir.exists():
        print(f"  SKIP {env_name}: checkpoint dir {policy} not found")
        return False

    # Check dataset exists
    ds_path = CACHE_DIR / cfg["eval_dataset_file"]
    if not ds_path.exists():
        print(f"  SKIP {env_name}: eval dataset {cfg['eval_dataset_file']} not found")
        return False

    cmd = [
        sys.executable, "eval.py",
        f"--config-name={cfg['eval_config']}",
        f"policy={policy}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Evaluating {env_name}: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="AF-LeWM-lite experiment runner")
    parser.add_argument("--mode", choices=["train", "eval", "both", "status"],
                        default="status", help="What to run")
    parser.add_argument("--env", choices=list(ENVS.keys()) + ["all"], default="all",
                        help="Which environment")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained checkpoints for eval (default)")
    parser.add_argument("--trained", action="store_true",
                        help="Use locally trained checkpoints for eval")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max_epochs for training")
    parser.add_argument("--num-eval", type=int, default=None,
                        help="Override number of eval episodes")
    args = parser.parse_args()

    envs = list(ENVS.keys()) if args.env == "all" else [args.env]
    use_pretrained = not args.trained

    if args.mode == "status":
        check_status()
        return

    if args.mode in ("train", "both"):
        extra = []
        if args.epochs:
            extra.append(f"trainer.max_epochs={args.epochs}")
        for env in envs:
            run_train(env, extra_args=extra)

    if args.mode in ("eval", "both"):
        extra = []
        if args.num_eval:
            extra.append(f"eval.num_eval={args.num_eval}")
        for env in envs:
            run_eval(env, use_pretrained=use_pretrained, extra_args=extra)


if __name__ == "__main__":
    main()
