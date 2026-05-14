"""Legacy Stage 1/2 helper for the PushT AF-LeWM study."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

if os.name == "nt":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ROOT = Path(__file__).parent
CACHE_DIR = Path(os.environ["STABLEWM_HOME"])
PUSHT_DATASET = "pusht_expert_train.h5"
PUSHT_MIN_OFFICIAL_BYTES = 1_000_000_000

PUSHT_STUDY_RUNS = [
    {
        "name": "baseline",
        "train_config": "lewm_pusht_ablation",
        "output_model_name": "lewm_pusht_reliable",
    },
    {
        "name": "af_v1",
        "train_config": "aflewm_pusht_v1_ablation",
        "output_model_name": "aflewm_pusht_v1_reliable",
    },
    {
        "name": "af_v2",
        "train_config": "aflewm_pusht_v2_ablation",
        "output_model_name": "aflewm_pusht_v2_reliable",
    },
]


def study_runs(epoch: int = 10) -> list[dict[str, str]]:
    runs = []
    for run in PUSHT_STUDY_RUNS:
        current = dict(run)
        current["policy"] = (
            "runs/pusht_expert_train/"
            f"{current['output_model_name']}/{current['output_model_name']}_epoch_{epoch}"
        )
        runs.append(current)
    return runs


def require_pusht_dataset() -> Path | None:
    dataset_path = CACHE_DIR / PUSHT_DATASET
    if not dataset_path.exists():
        print(f"  SKIP: dataset not found: {dataset_path}")
        return None
    size = dataset_path.stat().st_size
    if size < PUSHT_MIN_OFFICIAL_BYTES:
        print(
            f"  SKIP: dataset is too small for the official study: "
            f"{dataset_path} ({size / 1e9:.2f} GB)"
        )
        return None
    return dataset_path


def check_status() -> bool:
    print("=" * 70)
    print("AF-LeWM Legacy PushT Status")
    print(f"STABLEWM_HOME: {CACHE_DIR}")
    print("=" * 70)

    ok = True
    dataset_path = CACHE_DIR / PUSHT_DATASET
    if not dataset_path.exists():
        print(f"  PushT dataset:  [MISSING] {PUSHT_DATASET}")
        ok = False
    else:
        size_gb = dataset_path.stat().st_size / 1e9
        status = "OK" if dataset_path.stat().st_size >= PUSHT_MIN_OFFICIAL_BYTES else "PLACEHOLDER"
        print(f"  PushT dataset:  [{status}] {PUSHT_DATASET} ({size_gb:.2f} GB)")
        ok = ok and status == "OK"

    archive = CACHE_DIR / "pusht_expert_train.h5.zst"
    if archive.exists() and not dataset_path.exists():
        print(f"  Pending extract: {archive.name} ({archive.stat().st_size / 1e9:.2f} GB)")

    for run in study_runs():
        ckpt_path = CACHE_DIR / f"{run['policy']}_object.ckpt"
        status = "OK" if ckpt_path.exists() else "MISSING"
        print(f"  Study {run['name']}: [{status}] {run['policy']}")
        ok = ok and ckpt_path.exists()

    return ok


def run_train(extra_args: list[str] | None = None) -> bool:
    if require_pusht_dataset() is None:
        print("  SKIP training: official PushT dataset is not ready.")
        return False

    ok = True
    for run in study_runs():
        cmd = [
            sys.executable,
            "train.py",
            f"--config-name={run['train_config']}",
            "data=pusht",
            f"output_model_name={run['output_model_name']}",
            "wandb.enabled=False",
        ]
        if extra_args:
            cmd.extend(extra_args)

        print(f"\n{'=' * 60}")
        print(f"Training PushT/{run['name']}: {' '.join(cmd)}")
        print(f"{'=' * 60}")
        result = subprocess.run(cmd, cwd=ROOT)
        ok = ok and result.returncode == 0
        if result.returncode != 0:
            break
    return ok


def _eval_output_filename(base_filename: str, num_eval: int | None) -> str:
    if num_eval in (None, 50):
        return base_filename
    path = Path(base_filename)
    return f"{path.stem}_num{num_eval}{path.suffix}"


def run_eval(extra_args: list[str] | None = None, num_eval: int | None = None, epoch: int = 10) -> bool:
    if require_pusht_dataset() is None:
        print("  SKIP evaluation: official PushT dataset is not ready.")
        return False

    runs = study_runs(epoch=epoch)
    missing = [
        run["policy"]
        for run in runs
        if not (CACHE_DIR / f"{run['policy']}_object.ckpt").exists()
    ]
    if missing:
        print("  SKIP evaluation: reliable study checkpoints are incomplete.")
        for policy in missing:
            print(f"    missing: {policy}")
        return False

    ok = True
    for run in runs:
        for seed, output_filename in [(None, "pusht_results.txt"), (43, "pusht_results_seed43.txt")]:
            output_filename = _eval_output_filename(output_filename, num_eval)
            cmd = [
                sys.executable,
                "eval.py",
                "--config-name=pusht.yaml",
                f"policy={run['policy']}",
                f"output.filename={output_filename}",
            ]
            if seed is not None:
                cmd.append(f"seed={seed}")
            if extra_args:
                cmd.extend(extra_args)

            seed_label = f" seed={seed}" if seed is not None else ""
            print(f"\n{'=' * 60}")
            print(f"Evaluating PushT/{run['name']}{seed_label}: {' '.join(cmd)}")
            print(f"{'=' * 60}")
            result = subprocess.run(cmd, cwd=ROOT)
            ok = ok and result.returncode == 0
            if result.returncode != 0:
                return False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy AF-LeWM PushT runner")
    parser.add_argument("--mode", choices=["train", "eval", "both", "status"], default="status")
    parser.add_argument("--env", choices=["pusht"], default="pusht", help="Kept for CLI compatibility")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--num-eval", type=int, default=None, help="Override eval episodes")
    args = parser.parse_args()

    if args.mode == "status":
        check_status()
        return

    if args.mode in ("train", "both"):
        extra = []
        if args.epochs:
            extra.append(f"trainer.max_epochs={args.epochs}")
        if not run_train(extra):
            sys.exit(1)

    if args.mode in ("eval", "both"):
        extra = []
        if args.num_eval:
            extra.append(f"eval.num_eval={args.num_eval}")
        eval_epoch = args.epochs or 10
        if not run_eval(extra, num_eval=args.num_eval, epoch=eval_epoch):
            sys.exit(1)


if __name__ == "__main__":
    main()
