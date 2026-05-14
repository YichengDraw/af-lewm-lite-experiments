from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

ROOT = Path(__file__).resolve().parents[1]
STABLEWM_HOME = Path(os.environ["STABLEWM_HOME"]).expanduser()
RUNS_ROOT = STABLEWM_HOME / "runs" / "pusht_expert_train"
REPORT_DIR = ROOT / "report"
PUSHT_DATASET = STABLEWM_HOME / "pusht_expert_train.h5"
PUSHT_MIN_OFFICIAL_BYTES = 1_000_000_000
DEFAULT_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PYTHON_EXE = os.environ.get("AFLEWM_PYTHON") or (
    str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else sys.executable
)


@dataclass(frozen=True)
class Variant:
    variant_id: str
    train_config: str
    output_model_name: str
    description: str
    overrides: tuple[str, ...] = ()
    reuse_existing: bool = False


STAGE1_VARIANTS: tuple[Variant, ...] = (
    Variant(
        "baseline",
        "lewm_pusht_ablation",
        "lewm_pusht_reliable",
        "Baseline LeWM single dynamics latent.",
        reuse_existing=True,
    ),
    Variant(
        "v1_current",
        "aflewm_pusht_v1_ablation",
        "aflewm_pusht_v1_reliable",
        "Current v1: appearance projector + invariance + independence.",
        reuse_existing=True,
    ),
    Variant(
        "v1_inv_only",
        "aflewm_pusht_v1_ablation",
        "aflewm_pusht_v1_inv_only",
        "v1 without cross-cov independence.",
        ("aflwm.loss.independence_weight=0.0",),
    ),
    Variant(
        "v1_indep_only",
        "aflewm_pusht_v1_ablation",
        "aflewm_pusht_v1_indep_only",
        "v1 without augmentation invariance.",
        ("aflwm.loss.invariance_weight=0.0",),
    ),
    Variant(
        "v1_seq_only",
        "aflewm_pusht_v1_ablation",
        "aflewm_pusht_v1_seq_only",
        "v1 with sequence-consistent appearance augmentations.",
        ("+aflwm.augment.consistent_across_time=True",),
    ),
    Variant(
        "v1_seq_stopgrad",
        "aflewm_pusht_v1_ablation",
        "aflewm_pusht_v1_seq_stopgrad",
        "v1 with sequence-consistent augmentations and stop-grad invariance target.",
        (
            "+aflwm.augment.consistent_across_time=True",
            "+aflwm.invariance.stopgrad_target=True",
        ),
    ),
    Variant(
        "v2_app_nuisance_only",
        "aflewm_pusht_v2_ablation",
        "aflewm_pusht_v2_app_nuisance_only",
        "v2 without dynamics-side adversarial nuisance loss.",
        ("aflwm.loss.dynamics_nuisance_weight=0.0",),
    ),
    Variant(
        "v2_weak_grl",
        "aflewm_pusht_v2_ablation",
        "aflewm_pusht_v2_weak_grl",
        "v2 with lighter dynamics nuisance pressure.",
        (
            "aflwm.loss.dynamics_nuisance_weight=0.01",
            "aflwm.loss.grl_lambda=0.5",
        ),
    ),
    Variant(
        "v2_current",
        "aflewm_pusht_v2_ablation",
        "aflewm_pusht_v2_reliable",
        "Current v2: sequence-consistent nuisance shaping + full GRL.",
        reuse_existing=True,
    ),
    Variant(
        "v2_grl_warmup",
        "aflewm_pusht_v2_ablation",
        "aflewm_pusht_v2_grl_warmup",
        "Current v2 with GRL lambda linearly warmed over the first 30% of training steps.",
        ("+aflwm.loss.grl_warmup_frac=0.3",),
    ),
)


def require_dataset() -> None:
    if not PUSHT_DATASET.exists():
        raise FileNotFoundError(f"Missing official PushT dataset: {PUSHT_DATASET}")
    if PUSHT_DATASET.stat().st_size < PUSHT_MIN_OFFICIAL_BYTES:
        raise RuntimeError(f"PushT dataset is too small to be the official file: {PUSHT_DATASET}")


def variants_by_id() -> dict[str, Variant]:
    return {variant.variant_id: variant for variant in STAGE1_VARIANTS}


def selected_variants(ids: list[str] | None) -> list[Variant]:
    variants = variants_by_id()
    if not ids:
        return list(STAGE1_VARIANTS)
    missing = [variant_id for variant_id in ids if variant_id not in variants]
    if missing:
        raise KeyError(f"Unknown ablation variant(s): {missing}")
    return [variants[variant_id] for variant_id in ids]


def run_dir(output_model_name: str) -> Path:
    return RUNS_ROOT / output_model_name


def policy_path(output_model_name: str, epoch: int) -> str:
    return f"runs/pusht_expert_train/{output_model_name}/{output_model_name}_epoch_{epoch}"


def checkpoint_path(output_model_name: str, epoch: int) -> Path:
    return STABLEWM_HOME / f"{policy_path(output_model_name, epoch)}_object.ckpt"


def diagnostics_filename(epoch: int | None = None) -> str:
    return f"latent_diagnostics_epoch{epoch}.json" if epoch is not None else "latent_diagnostics.json"


def result_filename(seed: int, num_eval: int, epoch: int | None = None) -> str:
    if seed == 42:
        base = "pusht_results.txt"
    else:
        base = f"pusht_results_seed{seed}.txt"
    suffixes = []
    if epoch is not None:
        suffixes.append(f"epoch{epoch}")
    if num_eval != 50:
        suffixes.append(f"num{num_eval}")
    if not suffixes:
        return base
    path = Path(base)
    return f"{path.stem}_{'_'.join(suffixes)}{path.suffix}"


def run_command(cmd: list[str], *, dry_run: bool) -> int:
    print(" ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def train_variant(
    variant: Variant,
    *,
    dry_run: bool,
    force: bool,
    smoke: bool,
    seed: int | None = None,
    output_suffix: str = "",
    max_epochs: int | None = None,
    object_epoch_interval: int | None = None,
) -> bool:
    require_dataset()
    output_name = f"{variant.output_model_name}{output_suffix}"
    epoch = max_epochs or (1 if smoke else 10)
    ckpt = checkpoint_path(output_name, epoch)
    if ckpt.exists() and not force:
        print(f"SKIP train {variant.variant_id}: checkpoint exists: {ckpt}")
        return True

    cmd = [
        PYTHON_EXE,
        "train.py",
        f"--config-name={variant.train_config}",
        "data=pusht",
        f"output_model_name={output_name}",
        "wandb.enabled=False",
        *variant.overrides,
    ]
    if seed is not None:
        cmd.append(f"seed={seed}")
    if max_epochs is not None:
        cmd.append(f"trainer.max_epochs={max_epochs}")
    if object_epoch_interval is not None:
        cmd.append(f"+object_epoch_interval={object_epoch_interval}")
    if smoke:
        cmd.extend(
            [
                "trainer.max_epochs=1",
                "trainer.limit_train_batches=2",
                "trainer.limit_val_batches=1",
                "loader.batch_size=2",
            ]
        )
    return run_command(cmd, dry_run=dry_run) == 0


def eval_variant(
    variant: Variant,
    *,
    dry_run: bool,
    force: bool,
    smoke: bool,
    seeds: list[int],
    num_eval: int,
    epoch: int,
    output_suffix: str = "",
    result_epoch: int | None = None,
) -> bool:
    require_dataset()
    output_name = f"{variant.output_model_name}{output_suffix}"
    ckpt = checkpoint_path(output_name, epoch)
    if not ckpt.exists() and not dry_run:
        print(f"SKIP eval {variant.variant_id}: missing checkpoint: {ckpt}")
        return False

    ok = True
    for seed in seeds:
        filename = result_filename(seed, 2 if smoke else num_eval, result_epoch)
        output_path = run_dir(output_name) / filename
        if output_path.exists() and not force:
            print(f"SKIP eval {variant.variant_id} seed={seed}: result exists: {output_path}")
            continue
        cmd = [
            PYTHON_EXE,
            "eval.py",
            "--config-name=pusht.yaml",
            f"policy={policy_path(output_name, epoch)}",
            f"seed={seed}",
            f"output.filename={filename}",
        ]
        if smoke:
            cmd.extend(
                [
                    "eval.num_eval=2",
                    "solver.num_samples=4",
                    "solver.n_steps=2",
                    "solver.topk=2",
                ]
            )
        elif num_eval != 50:
            cmd.append(f"eval.num_eval={num_eval}")
        ok = run_command(cmd, dry_run=dry_run) == 0 and ok
    return ok


def diagnose_variant(
    variant: Variant,
    *,
    dry_run: bool,
    force: bool,
    epoch: int,
    output_suffix: str = "",
    num_batches: int = 32,
    result_epoch: int | None = None,
) -> bool:
    output_name = f"{variant.output_model_name}{output_suffix}"
    ckpt = checkpoint_path(output_name, epoch)
    if not ckpt.exists() and not dry_run:
        print(f"SKIP diagnose {variant.variant_id}: missing checkpoint: {ckpt}")
        return False

    output_path = run_dir(output_name) / diagnostics_filename(result_epoch)
    if output_path.exists() and not force:
        print(f"SKIP diagnose {variant.variant_id}: diagnostics exist: {output_path}")
        return True

    cmd = [
        PYTHON_EXE,
        "tools/diagnose_latents.py",
        f"--policy={policy_path(output_name, epoch)}",
        f"--num-batches={num_batches}",
        f"--output={output_path}",
    ]
    return run_command(cmd, dry_run=dry_run) == 0


def parse_eval_result(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    text = path.read_text()
    metrics_match = re.search(r"^metrics_json: (.+)$", text, re.M)
    time_match = re.search(r"evaluation_time: ([0-9.]+) seconds", text)
    if not metrics_match:
        return None
    metrics = json.loads(metrics_match.group(1))
    flags = metrics.get("episode_successes", [])
    return {
        "successes": sum(bool(flag) for flag in flags),
        "episodes": len(flags),
        "success_rate": metrics.get("success_rate"),
        "evaluation_time_seconds": float(time_match.group(1)) if time_match else None,
        "solver_batch_size": metrics.get("solver_batch_size"),
        "normalizer_scope": metrics.get("normalizer_metadata", {}).get("normalizer_scope"),
    }


def latest_val_metrics(output_model_name: str) -> dict[str, object]:
    metrics_path = run_dir(output_model_name) / "metrics" / "metrics.csv"
    if not metrics_path.exists():
        return {}
    rows = list(csv.DictReader(metrics_path.open(newline="")))
    val_rows = [row for row in rows if row.get("validate/pred_loss_epoch")]
    if not val_rows:
        return {}
    row = val_rows[-1]
    return {
        "validate_pred_loss_epoch": row.get("validate/pred_loss_epoch"),
        "validate_sigreg_loss_epoch": row.get("validate/sigreg_loss_epoch"),
        "validate_appearance_inv_loss_epoch": row.get("validate/appearance_inv_loss_epoch"),
        "validate_appearance_indep_loss_epoch": row.get("validate/appearance_indep_loss_epoch"),
        "validate_appearance_nuisance_loss_epoch": row.get("validate/appearance_nuisance_loss_epoch"),
        "validate_dynamics_nuisance_loss_epoch": row.get("validate/dynamics_nuisance_loss_epoch"),
        "validate_grl_lambda_loss_epoch": row.get("validate/grl_lambda_loss_epoch"),
    }


def load_diagnostics(output_model_name: str, result_epoch: int | None = None) -> dict[str, object]:
    path = run_dir(output_model_name) / diagnostics_filename(result_epoch)
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("metrics", {})


def clean_report_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def clean_report_row(row: dict[str, object]) -> dict[str, object]:
    return {key: clean_report_value(value) for key, value in row.items()}


def report_name_suffix(stage: str, epoch: int, smoke: bool) -> str:
    stage_label = f"{stage}_smoke" if smoke else stage
    if stage == "stage2":
        return f"{stage_label}_epoch{epoch}_summary"
    return f"{stage_label}_summary"


def train_seed_from_suffix(suffix: str) -> int | str:
    match = re.search(r"_s2_seed(\d+)", suffix)
    return int(match.group(1)) if match else ""


def write_report(
    variants: list[Variant],
    *,
    seeds: list[int],
    num_eval: int,
    epoch: int,
    stage: str,
    output_suffixes: list[str] | None = None,
    result_epoch: int | None = None,
    smoke: bool = False,
) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    rows = []
    suffixes = output_suffixes or [""]
    for variant in variants:
        for suffix in suffixes:
            output_name = f"{variant.output_model_name}{suffix}"
            seed_results = {
                seed: parse_eval_result(run_dir(output_name) / result_filename(seed, num_eval, result_epoch))
                for seed in seeds
            }
            successes = sum(result["successes"] for result in seed_results.values() if result)
            episodes = sum(result["episodes"] for result in seed_results.values() if result)
            row = {
                "stage": stage,
                "variant_id": variant.variant_id,
                "train_seed": train_seed_from_suffix(suffix),
                "output_suffix": suffix,
                "output_model_name": output_name,
                "description": variant.description,
                "checkpoint_exists": checkpoint_path(output_name, epoch).exists(),
                "aggregate_successes": successes,
                "aggregate_episodes": episodes,
                "aggregate_success_percent": 100.0 * successes / episodes if episodes else "",
            }
            for seed, result in seed_results.items():
                row[f"seed{seed}_successes"] = result["successes"] if result else ""
                row[f"seed{seed}_episodes"] = result["episodes"] if result else ""
                row[f"seed{seed}_success_rate"] = result["success_rate"] if result else ""
            row.update(latest_val_metrics(output_name))
            for key, value in load_diagnostics(output_name, result_epoch).items():
                row[f"diag_{key}"] = value
            rows.append(clean_report_row(row))

    report_suffix = report_name_suffix(stage, epoch, smoke)
    json_path = REPORT_DIR / f"pusht_ablation_{report_suffix}.json"
    csv_path = REPORT_DIR / f"pusht_ablation_{report_suffix}.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def status(
    variants: list[Variant],
    *,
    epoch: int,
    seeds: list[int],
    num_eval: int,
    output_suffixes: list[str] | None = None,
    result_epoch: int | None = None,
) -> None:
    print(f"STABLEWM_HOME={STABLEWM_HOME}")
    print(f"dataset={'OK' if PUSHT_DATASET.exists() and PUSHT_DATASET.stat().st_size >= PUSHT_MIN_OFFICIAL_BYTES else 'MISSING'} {PUSHT_DATASET}")
    suffixes = output_suffixes or [""]
    for variant in variants:
        for suffix in suffixes:
            output_name = f"{variant.output_model_name}{suffix}"
            ckpt = checkpoint_path(output_name, epoch)
            result_bits = []
            for seed in seeds:
                result_path = run_dir(output_name) / result_filename(seed, num_eval, result_epoch)
                result_bits.append(f"seed{seed}={'OK' if result_path.exists() else 'MISSING'}")
            diag_path = run_dir(output_name) / diagnostics_filename(result_epoch)
            print(
                f"{variant.variant_id + suffix:34s} ckpt={'OK' if ckpt.exists() else 'MISSING'} "
                f"diag={'OK' if diag_path.exists() else 'MISSING'} "
                + " ".join(result_bits)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PushT AF-LeWM ablations.")
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--mode", choices=["status", "train", "eval", "diagnose", "report", "all"], default="status")
    parser.add_argument("--ids", nargs="*", default=None, help="Stage 1 variant ids to run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--eval-epoch", type=int, default=None)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--eval-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--stage2-train-seeds", type=int, nargs="*", default=[3072, 3073])
    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--stage2-object-interval", type=int, default=5)
    parser.add_argument("--diagnostic-batches", type=int, default=32)
    args = parser.parse_args()

    variants = selected_variants(args.ids)
    effective_num_eval = 2 if args.smoke else args.num_eval
    if args.stage == "stage2":
        base_suffixes = [f"_s2_seed{seed}" for seed in args.stage2_train_seeds]
        suffixes = [f"{suffix}_smoke" for suffix in base_suffixes] if args.smoke else base_suffixes
        train_jobs = [
            (variant, seed, f"{suffix}_smoke" if args.smoke else suffix)
            for variant in variants
            for seed, suffix in zip(args.stage2_train_seeds, base_suffixes)
        ]
        eval_epoch = 1 if args.smoke else (args.eval_epoch or args.stage2_epochs)
        train_epochs = 1 if args.smoke else args.stage2_epochs
        eval_seeds = args.eval_seeds if args.eval_seeds is not None else [42, 43, 44, 45]
        result_epoch = eval_epoch
    else:
        suffixes = ["_smoke"] if args.smoke else [""]
        train_jobs = [(variant, None, "_smoke" if args.smoke else "") for variant in variants]
        eval_epoch = 1 if args.smoke else (args.eval_epoch or args.epoch)
        train_epochs = None
        eval_seeds = args.eval_seeds if args.eval_seeds is not None else [42, 43]
        result_epoch = None

    if args.mode == "status":
        status(
            variants,
            epoch=eval_epoch,
            seeds=eval_seeds,
            num_eval=effective_num_eval,
            output_suffixes=suffixes,
            result_epoch=result_epoch,
        )
        return

    ok = True
    if args.mode in ("train", "all"):
        for variant, seed, suffix in train_jobs:
            ok = train_variant(
                variant,
                dry_run=args.dry_run,
                force=args.force,
                smoke=args.smoke,
                seed=seed,
                output_suffix=suffix,
                max_epochs=train_epochs,
                object_epoch_interval=args.stage2_object_interval if args.stage == "stage2" else None,
            ) and ok
            if not ok:
                sys.exit(1)
    if args.mode in ("eval", "all"):
        for suffix in suffixes:
            for variant in variants:
                ok = eval_variant(
                    variant,
                    dry_run=args.dry_run,
                    force=args.force,
                    smoke=args.smoke,
                    seeds=eval_seeds,
                    num_eval=effective_num_eval,
                    epoch=eval_epoch,
                    output_suffix=suffix,
                    result_epoch=result_epoch,
                ) and ok
                if not ok:
                    sys.exit(1)
    if args.mode in ("diagnose", "all"):
        for suffix in suffixes:
            for variant in variants:
                ok = diagnose_variant(
                    variant,
                    dry_run=args.dry_run,
                    force=args.force,
                    epoch=eval_epoch,
                    output_suffix=suffix,
                    num_batches=args.diagnostic_batches,
                    result_epoch=result_epoch,
                ) and ok
                if not ok:
                    sys.exit(1)
    if args.mode in ("report", "all"):
        if args.dry_run:
            report_suffix = report_name_suffix(args.stage, eval_epoch, args.smoke)
            print(f"DRY RUN: would write report/pusht_ablation_{report_suffix}.[json,csv]")
        else:
            write_report(
                variants,
                seeds=eval_seeds,
                num_eval=effective_num_eval,
                epoch=eval_epoch,
                stage=args.stage,
                output_suffixes=suffixes,
                result_epoch=result_epoch,
                smoke=args.smoke,
            )


if __name__ == "__main__":
    main()
