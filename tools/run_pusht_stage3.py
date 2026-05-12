"""Official-aligned PushT Stage 3 runner for baseline vs AF-LeWM v1.

This runner is intentionally separate from the earlier short-budget ablation
runner. Stage 3 needs locked eval manifests, checkpoint-by-checkpoint closed-loop
validation, W&B-aware launches, and paired statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

ROOT = Path(__file__).resolve().parents[1]
STABLEWM_HOME = Path(os.environ["STABLEWM_HOME"]).expanduser()
RUNS_ROOT = STABLEWM_HOME / "runs" / "pusht_expert_train"
REPORT_DIR = ROOT / "report"
MANIFEST_DIR = REPORT_DIR / "stage3_manifests"
PUSHT_DATASET = STABLEWM_HOME / "pusht_expert_train.h5"
DEFAULT_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PYTHON_EXE = os.environ.get("AFLEWM_PYTHON") or (
    str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else sys.executable
)

SPLIT_SEED = 9001
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
HISTORY_SPAN = 15
GOAL_OFFSET_STEPS = 25
DEFAULT_TRAIN_SEEDS = (3072, 3073, 3074, 3075, 3076)
DEFAULT_EVAL_EPOCHS = tuple(range(5, 101, 5))


@dataclass(frozen=True)
class Variant:
    variant_id: str
    train_config: str
    output_model_name: str
    description: str


VARIANTS = {
    "baseline": Variant(
        "baseline",
        "lewm_pusht_stage3",
        "lewm_pusht_stage3",
        "Official-aligned LeWM baseline.",
    ),
    "v1_current": Variant(
        "v1_current",
        "aflewm_pusht_v1_stage3",
        "aflewm_pusht_v1_stage3",
        "AF-LeWM v1: appearance projector with dynamics invariance and cross-cov independence.",
    ),
}


def run_command(cmd: list[str], *, dry_run: bool) -> int:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def require_dataset() -> None:
    if not PUSHT_DATASET.exists():
        raise FileNotFoundError(f"Missing PushT dataset: {PUSHT_DATASET}")
    if PUSHT_DATASET.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"PushT dataset is too small to be official: {PUSHT_DATASET}")


def output_name(
    variant: Variant,
    train_seed: int,
    smoke: bool = False,
    run_label: str = "",
) -> str:
    label = f"_{run_label}" if run_label else ""
    suffix = f"_s3_seed{train_seed}"
    if smoke:
        suffix += "_smoke"
    return f"{variant.output_model_name}{label}{suffix}"


def run_dir(name: str) -> Path:
    return RUNS_ROOT / name


def policy_path(name: str, epoch: int) -> str:
    return f"runs/pusht_expert_train/{name}/{name}_epoch_{epoch}"


def checkpoint_path(name: str, epoch: int) -> Path:
    return STABLEWM_HOME / f"{policy_path(name, epoch)}_object.ckpt"


def weights_checkpoint_path(name: str) -> Path:
    return run_dir(name) / f"{name}_weights.ckpt"


def eval_filename(split: str, epoch: int, num_eval: int) -> str:
    return f"stage3_{split}_epoch{epoch}_num{num_eval}.txt"


def manifest_path(split: str, num_eval: int, seed: int) -> Path:
    return MANIFEST_DIR / f"pusht_stage3_{split}_n{num_eval}_seed{seed}.json"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def selected_variants(ids: list[str] | None) -> list[Variant]:
    if not ids:
        return [VARIANTS["baseline"], VARIANTS["v1_current"]]
    missing = [variant_id for variant_id in ids if variant_id not in VARIANTS]
    if missing:
        raise KeyError(f"Unknown Stage 3 variant ids: {missing}")
    return [VARIANTS[variant_id] for variant_id in ids]


def episode_splits(num_episodes: int) -> tuple[list[int], list[int], list[int]]:
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(num_episodes, generator=generator).tolist()
    n_train = int(num_episodes * TRAIN_SPLIT)
    n_val = int(num_episodes * VAL_SPLIT)
    train = sorted(int(idx) for idx in perm[:n_train])
    val = sorted(int(idx) for idx in perm[n_train : n_train + n_val])
    test = sorted(int(idx) for idx in perm[n_train + n_val :])
    if not train or not val or not test:
        raise RuntimeError("Stage 3 split produced an empty train/val/test split")
    return train, val, test


def _load_h5_arrays() -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(PUSHT_DATASET, "r") as h5:
        return np.asarray(h5["ep_len"]), np.asarray(h5["ep_offset"])


def make_manifest(split: str, num_eval: int, seed: int, *, force: bool = False) -> Path:
    require_dataset()
    out_path = manifest_path(split, num_eval, seed)
    if out_path.exists() and not force:
        print(f"SKIP manifest: {out_path}")
        return out_path

    ep_len, ep_offset = _load_h5_arrays()
    _, val_episodes, test_episodes = episode_splits(len(ep_len))
    split_episodes = {"val": val_episodes, "test": test_episodes}[split]
    rng = np.random.default_rng(seed)
    shuffled = list(split_episodes)
    rng.shuffle(shuffled)

    rows: list[dict[str, int]] = []
    for episode_idx in shuffled:
        lo = HISTORY_SPAN - 1
        hi = int(ep_len[episode_idx]) - GOAL_OFFSET_STEPS - 1
        if hi < lo:
            continue
        start_step = int(rng.integers(lo, hi + 1))
        rows.append(
            {
                "row_index": int(ep_offset[episode_idx] + start_step),
                "episode_idx": int(episode_idx),
                "start_step": start_step,
            }
        )
        if len(rows) >= num_eval:
            break
    if len(rows) < num_eval:
        raise RuntimeError(
            f"Only generated {len(rows)} starts for {split}, requested {num_eval}"
        )
    rows = sorted(rows, key=lambda row: row["row_index"])
    payload = {
        "manifest_type": "pusht_stage3_eval_manifest",
        "split": split,
        "num_eval": num_eval,
        "seed": seed,
        "split_seed": SPLIT_SEED,
        "train_split": TRAIN_SPLIT,
        "val_split": VAL_SPLIT,
        "test_split": TEST_SPLIT,
        "history_span": HISTORY_SPAN,
        "goal_offset_steps": GOAL_OFFSET_STEPS,
        "dataset": str(PUSHT_DATASET),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
    return out_path


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_chunk_manifest(
    base_manifest_path: Path,
    base_manifest: dict[str, Any],
    *,
    chunk_index: int,
    chunk_count: int,
    start: int,
    end: int,
    force: bool,
) -> Path:
    rows = base_manifest["rows"][start:end]
    chunk_dir = MANIFEST_DIR / "chunks"
    chunk_path = chunk_dir / (
        f"{base_manifest_path.stem}_chunk{chunk_index:03d}_of{chunk_count:03d}.json"
    )
    payload = {
        **{k: v for k, v in base_manifest.items() if k != "rows"},
        "num_eval": len(rows),
        "rows": rows,
        "parent_manifest_path": str(base_manifest_path),
        "parent_manifest_sha256": _sha256_json(base_manifest["rows"]),
        "parent_num_eval": int(base_manifest["num_eval"]),
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "chunk_start": start,
        "chunk_end": end,
    }
    if chunk_path.exists() and not force:
        existing = read_manifest(chunk_path)
        if existing.get("rows") == rows and existing.get("parent_manifest_sha256") == payload["parent_manifest_sha256"]:
            print(f"SKIP chunk manifest: {chunk_path}")
            return chunk_path
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {chunk_path}")
    return chunk_path


def ensure_manifests(*, force: bool = False, smoke: bool = False) -> dict[str, Path]:
    val_small = 2 if smoke else 100
    val_large = 2 if smoke else 500
    test_size = 2 if smoke else 1000
    return {
        "val_small": make_manifest("val", val_small, 9100, force=force),
        "val_large": make_manifest("val", val_large, 9101, force=force),
        "test": make_manifest("test", test_size, 9200, force=force),
    }


def train_variant(
    variant: Variant,
    train_seed: int,
    *,
    dry_run: bool,
    force: bool,
    smoke: bool,
    wandb: bool,
    batch_size: int | None,
    target_epoch: int | None = None,
    resume: bool | None = None,
    run_label: str = "",
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    log_every_n_steps: int | None = None,
) -> bool:
    require_dataset()
    name = output_name(variant, train_seed, smoke, run_label)
    epoch = target_epoch if target_epoch is not None else (1 if smoke else 100)
    resume_enabled = weights_checkpoint_path(name).exists() if resume is None else resume
    ckpt = checkpoint_path(name, epoch)
    if ckpt.exists() and not force:
        print(f"SKIP train {name}: checkpoint exists {ckpt}")
        return True

    cmd = [
        PYTHON_EXE,
        "train.py",
        f"--config-name={variant.train_config}",
        "data=pusht",
        f"output_model_name={name}",
        f"subdir=runs/pusht_expert_train/{name}",
        f"seed={train_seed}",
        f"split_seed={SPLIT_SEED}",
        "train_split=0.8",
        "val_split=0.1",
        "test_split=0.1",
        "object_epoch_interval=5",
        "save_weights=True",
        f"trainer.max_epochs={epoch}",
        f"resume={str(resume_enabled)}",
        f"wandb.enabled={str(wandb and not smoke)}",
    ]
    if wandb and not smoke:
        cmd.extend(
            [
                f"wandb.config.name={name}",
                f"wandb.config.id={name}",
                "+wandb.config.group=pusht-stage3-v1",
            ]
        )
    if batch_size is not None and not smoke:
        cmd.append(f"loader.batch_size={batch_size}")
    if limit_train_batches is not None and not smoke:
        cmd.append(f"+trainer.limit_train_batches={limit_train_batches}")
    if limit_val_batches is not None and not smoke:
        cmd.append(f"+trainer.limit_val_batches={limit_val_batches}")
    if log_every_n_steps is not None and not smoke:
        cmd.append(f"+trainer.log_every_n_steps={log_every_n_steps}")
    if smoke:
        cmd.extend(
            [
                "+trainer.limit_train_batches=2",
                "+trainer.limit_val_batches=1",
                "loader.batch_size=2",
                "loader.num_workers=0",
                "num_workers=0",
                "loader.persistent_workers=False",
                "loader.prefetch_factor=null",
            ]
        )
    return run_command(cmd, dry_run=dry_run) == 0


def eval_variant(
    variant: Variant,
    train_seed: int,
    *,
    split: str,
    manifest: Path,
    epoch: int,
    dry_run: bool,
    force: bool,
    smoke: bool,
    num_samples: int | None = None,
    n_steps: int | None = None,
    solver_batch_size: int | None = None,
    test_chunk_size: int | None = None,
    output_filename: str | None = None,
    run_label: str = "",
    retries: int = 0,
) -> bool:
    name = output_name(variant, train_seed, smoke, run_label)
    ckpt = checkpoint_path(name, epoch)
    if not ckpt.exists() and not dry_run:
        print(f"SKIP eval {name} epoch={epoch}: missing {ckpt}")
        return False
    manifest_payload = read_manifest(manifest) if manifest.exists() else None
    num_eval = int(manifest_payload["num_eval"]) if manifest_payload else (2 if smoke else 100)
    filename = output_filename or eval_filename(split, epoch, num_eval)
    out_path = run_dir(name) / filename
    if out_path.exists() and not force:
        print(f"SKIP eval {name} {split} epoch={epoch}: {out_path}")
        return True
    if (
        split == "test"
        and manifest_payload is not None
        and test_chunk_size is not None
        and test_chunk_size > 0
        and num_eval > test_chunk_size
        and output_filename is None
    ):
        return eval_variant_chunked(
            variant,
            train_seed,
            manifest=manifest,
            manifest_payload=manifest_payload,
            epoch=epoch,
            dry_run=dry_run,
            force=force,
            smoke=smoke,
            num_samples=num_samples,
            n_steps=n_steps,
            solver_batch_size=solver_batch_size,
            test_chunk_size=test_chunk_size,
            run_label=run_label,
            retries=retries,
            final_output_path=out_path,
        )
    cmd = [
        PYTHON_EXE,
        "eval.py",
        "--config-name=pusht.yaml",
        f"policy={policy_path(name, epoch)}",
        f"eval.num_eval={num_eval}",
        f"eval.manifest_path={manifest}",
        f"output.filename={filename}",
    ]
    if smoke:
        cmd.extend(["solver.num_samples=4", "solver.n_steps=2", "solver.topk=2"])
    else:
        if num_samples is not None:
            cmd.append(f"solver.num_samples={num_samples}")
        if n_steps is not None:
            cmd.append(f"solver.n_steps={n_steps}")
        if solver_batch_size is not None:
            cmd.append(f"solver.batch_size={solver_batch_size}")
    for attempt in range(retries + 1):
        if run_command(cmd, dry_run=dry_run) == 0:
            return True
        if attempt < retries:
            delay = 30 * (attempt + 1)
            print(
                f"RETRY eval {name} {split} epoch={epoch}: "
                f"attempt {attempt + 1}/{retries} failed, sleeping {delay}s",
                flush=True,
            )
            if not dry_run:
                time.sleep(delay)
    return False


def load_eval_metrics(path: Path) -> tuple[dict[str, Any], float | None] | None:
    if not path.exists():
        return None
    text = path.read_text()
    metrics_match = re.search(r"^metrics_json: (.+)$", text, re.M)
    time_match = re.search(r"evaluation_time: ([0-9.]+) seconds", text)
    if not metrics_match:
        return None
    metrics = json.loads(metrics_match.group(1))
    elapsed = float(time_match.group(1)) if time_match else None
    return metrics, elapsed


def parse_eval_result(path: Path) -> dict[str, Any] | None:
    loaded = load_eval_metrics(path)
    if loaded is None:
        return None
    metrics, elapsed = loaded
    flags = [bool(flag) for flag in metrics.get("episode_successes", [])]
    return {
        "successes": int(sum(flags)),
        "episodes": len(flags),
        "success_percent": 100.0 * sum(flags) / len(flags) if flags else None,
        "flags": flags,
        "eval_row_indices": metrics.get("eval_row_indices", []),
        "eval_episodes": metrics.get("eval_episodes", []),
        "eval_start_idx": metrics.get("eval_start_idx", []),
        "manifest": metrics.get("eval_manifest"),
        "normalizer_scope": metrics.get("normalizer_metadata", {}).get("normalizer_scope"),
        "solver_batch_size": metrics.get("solver_batch_size"),
        "evaluation_time_seconds": elapsed,
    }


def eval_variant_chunked(
    variant: Variant,
    train_seed: int,
    *,
    manifest: Path,
    manifest_payload: dict[str, Any],
    epoch: int,
    dry_run: bool,
    force: bool,
    smoke: bool,
    num_samples: int | None,
    n_steps: int | None,
    solver_batch_size: int | None,
    test_chunk_size: int,
    run_label: str,
    retries: int,
    final_output_path: Path,
) -> bool:
    total = int(manifest_payload["num_eval"])
    chunk_count = math.ceil(total / test_chunk_size)
    name = output_name(variant, train_seed, smoke, run_label)
    chunk_paths: list[Path] = []

    for chunk_index, start in enumerate(range(0, total, test_chunk_size), start=1):
        end = min(start + test_chunk_size, total)
        chunk_manifest = write_chunk_manifest(
            manifest,
            manifest_payload,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            start=start,
            end=end,
            force=force,
        )
        chunk_num_eval = end - start
        chunk_filename = (
            f"stage3_test_epoch{epoch}_num{total}"
            f"_chunk{chunk_index:03d}_of{chunk_count:03d}_num{chunk_num_eval}.txt"
        )
        ok = eval_variant(
            variant,
            train_seed,
            split="test",
            manifest=chunk_manifest,
            epoch=epoch,
            dry_run=dry_run,
            force=force,
            smoke=smoke,
            num_samples=num_samples,
            n_steps=n_steps,
            solver_batch_size=solver_batch_size,
            test_chunk_size=None,
            output_filename=chunk_filename,
            run_label=run_label,
            retries=retries,
        )
        if not ok:
            return False
        chunk_paths.append(run_dir(name) / chunk_filename)

    if dry_run:
        print(f"DRY RUN: would aggregate {chunk_count} chunks into {final_output_path}")
        return True
    return aggregate_chunked_eval_result(
        manifest,
        manifest_payload,
        chunk_paths,
        final_output_path,
        epoch=epoch,
        test_chunk_size=test_chunk_size,
    )


def aggregate_chunked_eval_result(
    manifest_path_: Path,
    manifest_payload: dict[str, Any],
    chunk_paths: list[Path],
    out_path: Path,
    *,
    epoch: int,
    test_chunk_size: int,
) -> bool:
    flags: list[bool] = []
    seeds: list[Any] = []
    row_indices: list[int] = []
    episodes: list[int] = []
    start_idx: list[int] = []
    chunk_summaries: list[dict[str, Any]] = []
    elapsed_total = 0.0
    first_metrics: dict[str, Any] | None = None

    for chunk_path in chunk_paths:
        loaded = load_eval_metrics(chunk_path)
        if loaded is None:
            print(f"Missing chunk result for aggregation: {chunk_path}")
            return False
        metrics, elapsed = loaded
        if first_metrics is None:
            first_metrics = metrics
        chunk_flags = [bool(flag) for flag in metrics.get("episode_successes", [])]
        flags.extend(chunk_flags)
        seeds.extend(metrics.get("seeds") or [])
        row_indices.extend(int(value) for value in metrics.get("eval_row_indices", []))
        episodes.extend(int(value) for value in metrics.get("eval_episodes", []))
        start_idx.extend(int(value) for value in metrics.get("eval_start_idx", []))
        if elapsed is not None:
            elapsed_total += elapsed
        chunk_summaries.append(
            {
                "path": str(chunk_path),
                "episodes": len(chunk_flags),
                "successes": int(sum(chunk_flags)),
                "success_rate": 100.0 * sum(chunk_flags) / len(chunk_flags) if chunk_flags else None,
                "evaluation_time_seconds": elapsed,
            }
        )

    expected_rows = manifest_payload["rows"]
    expected_row_indices = [int(row["row_index"]) for row in expected_rows]
    expected_episodes = [int(row["episode_idx"]) for row in expected_rows]
    expected_start_idx = [int(row["start_step"]) for row in expected_rows]
    if row_indices != expected_row_indices:
        raise ValueError("Chunk aggregation row order does not match the base test manifest")
    if episodes != expected_episodes or start_idx != expected_start_idx:
        raise ValueError("Chunk aggregation episode/start metadata does not match the base test manifest")

    merged = dict(first_metrics or {})
    merged.update(
        {
            "success_rate": 100.0 * sum(flags) / len(flags) if flags else 0.0,
            "episode_successes": flags,
            "seeds": seeds,
            "eval_row_indices": row_indices,
            "eval_episodes": episodes,
            "eval_start_idx": start_idx,
            "eval_manifest": {
                "manifest_path": str(manifest_path_),
                "manifest_sha256": _sha256_json(expected_rows),
                "split": manifest_payload.get("split"),
                "size": len(expected_rows),
                "seed": manifest_payload.get("seed"),
            },
            "chunked_eval": {
                "chunk_size": test_chunk_size,
                "chunk_count": len(chunk_paths),
                "chunks": chunk_summaries,
            },
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("==== CONFIG ====\n")
        fh.write("chunked_stage3_eval: true\n")
        fh.write(f"epoch: {epoch}\n")
        fh.write(f"manifest_path: {manifest_path_}\n")
        fh.write(f"test_chunk_size: {test_chunk_size}\n")
        fh.write("==== RESULTS ====\n")
        fh.write(f"metrics: {merged}\n")
        fh.write(f"metrics_json: {json.dumps(merged, sort_keys=True)}\n")
        fh.write(f"evaluation_time: {elapsed_total} seconds\n")
    print(f"Wrote chunked aggregate: {out_path}")
    return True


def val_num_eval_for_manifest(*, smoke: bool, val_manifest_kind: str) -> int:
    if smoke:
        return 2
    return 500 if val_manifest_kind == "large" else 100


def terminal_epoch_for_run(epochs: list[int], *, smoke: bool) -> int:
    if smoke:
        return 1
    return max(epochs) if epochs else 100


def latest_val_loss(name: str) -> dict[str, str]:
    path = run_dir(name) / "metrics" / "metrics.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(newline="")))
    val_rows = [row for row in rows if row.get("validate/pred_loss_epoch")]
    if not val_rows:
        return {}
    row = val_rows[-1]
    return {
        "validate_pred_loss_epoch": row.get("validate/pred_loss_epoch", ""),
        "validate_loss_epoch": row.get("validate/loss_epoch", ""),
    }


def _best_epoch_for(name: str, epochs: list[int], num_eval: int) -> tuple[int | None, dict[str, Any] | None]:
    best_epoch = None
    best_result = None
    for epoch in epochs:
        result = parse_eval_result(run_dir(name) / eval_filename("val", epoch, num_eval))
        if result is None:
            continue
        if best_result is None or (result["success_percent"] or -1) > (best_result["success_percent"] or -1):
            best_epoch = epoch
            best_result = result
    return best_epoch, best_result


def paired_delta_percent(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    if a is None or b is None:
        return None
    if a["eval_row_indices"] != b["eval_row_indices"]:
        raise ValueError("Cannot compute paired delta: eval row indices differ")
    aflags = np.asarray(a["flags"], dtype=np.float32)
    bflags = np.asarray(b["flags"], dtype=np.float32)
    return float((bflags - aflags).mean() * 100.0)


def bootstrap_ci_delta(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    *,
    seed: int = 12345,
    samples: int = 5000,
) -> tuple[float | None, float | None]:
    if baseline is None or candidate is None:
        return None, None
    if baseline["eval_row_indices"] != candidate["eval_row_indices"]:
        raise ValueError("Cannot bootstrap paired delta: eval row indices differ")
    diffs = np.asarray(candidate["flags"], dtype=np.float32) - np.asarray(
        baseline["flags"], dtype=np.float32
    )
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float32)
    n = len(diffs)
    for idx in range(samples):
        means[idx] = diffs[rng.integers(0, n, size=n)].mean() * 100.0
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_report(
    variants: list[Variant],
    train_seeds: list[int],
    epochs: list[int],
    *,
    smoke: bool,
    run_label: str = "",
    val_manifest_kind: str = "small",
) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    val_num_eval = val_num_eval_for_manifest(
        smoke=smoke,
        val_manifest_kind=val_manifest_kind,
    )
    test_num_eval = 2 if smoke else 1000
    final_epoch = terminal_epoch_for_run(epochs, smoke=smoke)
    label = f"_{run_label}" if run_label else ""
    report_stem = "pusht_stage3_smoke" if smoke else f"pusht_stage3_v1{label}"

    val_curve_rows = []
    rows = []
    for variant in variants:
        for seed in train_seeds:
            name = output_name(variant, seed, smoke, run_label)
            for epoch in epochs:
                val_result = parse_eval_result(run_dir(name) / eval_filename("val", epoch, val_num_eval))
                val_curve_rows.append(
                    {
                        "variant_id": variant.variant_id,
                        "train_seed": seed,
                        "output_model_name": name,
                        "epoch": epoch,
                        "val_success_percent": val_result["success_percent"] if val_result else "",
                        "val_successes": val_result["successes"] if val_result else "",
                        "val_episodes": val_result["episodes"] if val_result else "",
                        "normalizer_scope": val_result["normalizer_scope"] if val_result else "",
                        "solver_batch_size": val_result["solver_batch_size"] if val_result else "",
                        "evaluation_time_seconds": val_result["evaluation_time_seconds"] if val_result else "",
                    }
                )
            best_epoch, best_val = _best_epoch_for(name, epochs, val_num_eval)
            test = None
            if best_epoch is not None:
                test = parse_eval_result(run_dir(name) / eval_filename("test", best_epoch, test_num_eval))
            row = {
                "variant_id": variant.variant_id,
                "train_seed": seed,
                "output_model_name": name,
                "best_val_epoch": best_epoch or "",
                "best_val_success_percent": best_val["success_percent"] if best_val else "",
                "test_success_percent": test["success_percent"] if test else "",
                "test_successes": test["successes"] if test else "",
                "test_episodes": test["episodes"] if test else "",
                "final_ckpt_exists": checkpoint_path(name, final_epoch).exists(),
            }
            row.update(latest_val_loss(name))
            rows.append(row)

    curve_csv_path = REPORT_DIR / f"{report_stem}_val_curve.csv"
    curve_json_path = curve_csv_path.with_suffix(".json")
    curve_json_path.write_text(json.dumps(val_curve_rows, indent=2, sort_keys=True))
    curve_fieldnames = sorted({key for row in val_curve_rows for key in row})
    with curve_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=curve_fieldnames)
        writer.writeheader()
        writer.writerows(val_curve_rows)

    csv_path = REPORT_DIR / f"{report_stem}_summary.csv"
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if {"baseline", "v1_current"}.issubset({variant.variant_id for variant in variants}):
        paired_rows = []
        for seed in train_seeds:
            baseline_name = output_name(VARIANTS["baseline"], seed, smoke, run_label)
            v1_name = output_name(VARIANTS["v1_current"], seed, smoke, run_label)
            baseline_best, _ = _best_epoch_for(baseline_name, epochs, val_num_eval)
            v1_best, _ = _best_epoch_for(v1_name, epochs, val_num_eval)
            if baseline_best is None or v1_best is None:
                continue
            baseline_test = parse_eval_result(
                run_dir(baseline_name) / eval_filename("test", baseline_best, test_num_eval)
            )
            v1_test = parse_eval_result(
                run_dir(v1_name) / eval_filename("test", v1_best, test_num_eval)
            )
            delta = paired_delta_percent(baseline_test, v1_test)
            ci_low, ci_high = bootstrap_ci_delta(baseline_test, v1_test)
            paired_rows.append(
                {
                    "train_seed": seed,
                    "baseline_best_epoch": baseline_best,
                    "v1_best_epoch": v1_best,
                    "baseline_test_success_percent": baseline_test["success_percent"] if baseline_test else "",
                    "v1_test_success_percent": v1_test["success_percent"] if v1_test else "",
                    "paired_delta_v1_minus_baseline_percent": delta if delta is not None else "",
                    "paired_delta_ci95_low": ci_low if ci_low is not None else "",
                    "paired_delta_ci95_high": ci_high if ci_high is not None else "",
                }
            )
        paired_csv = REPORT_DIR / (
            "pusht_stage3_smoke_paired.csv"
            if smoke
            else f"pusht_stage3_v1{label}_paired.csv"
        )
        paired_json = paired_csv.with_suffix(".json")
        paired_json.write_text(json.dumps(paired_rows, indent=2, sort_keys=True))
        if paired_rows:
            with paired_csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=sorted({k for row in paired_rows for k in row}))
                writer.writeheader()
                writer.writerows(paired_rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {curve_json_path}")
    print(f"Wrote {curve_csv_path}")


def status(
    variants: list[Variant],
    train_seeds: list[int],
    epochs: list[int],
    *,
    smoke: bool,
    run_label: str = "",
    val_manifest_kind: str = "small",
) -> None:
    print(f"STABLEWM_HOME={STABLEWM_HOME}")
    print(f"dataset={PUSHT_DATASET} exists={PUSHT_DATASET.exists()}")
    val_num_eval = val_num_eval_for_manifest(
        smoke=smoke,
        val_manifest_kind=val_manifest_kind,
    )
    for variant in variants:
        for seed in train_seeds:
            name = output_name(variant, seed, smoke, run_label)
            ckpts = [epoch for epoch in epochs if checkpoint_path(name, epoch).exists()]
            evals = [
                epoch
                for epoch in epochs
                if (run_dir(name) / eval_filename("val", epoch, val_num_eval)).exists()
            ]
            print(
                f"{name}: weights={weights_checkpoint_path(name).exists()} "
                f"ckpts={ckpts[-5:]} evals={evals[-5:]}"
            )


def parse_epochs(values: list[int] | None, smoke: bool) -> list[int]:
    if smoke:
        return [1]
    return values if values else list(DEFAULT_EVAL_EPOCHS)


def run_selected_test_evals(
    variants: list[Variant],
    train_seeds: list[int],
    epochs: list[int],
    *,
    manifests: dict[str, Path],
    dry_run: bool,
    force: bool,
    smoke: bool,
    num_samples: int | None,
    n_steps: int | None,
    solver_batch_size: int | None = None,
    test_chunk_size: int | None = None,
    run_label: str = "",
    val_manifest_kind: str = "small",
    eval_retries: int = 0,
) -> bool:
    val_num_eval = val_num_eval_for_manifest(
        smoke=smoke,
        val_manifest_kind=val_manifest_kind,
    )
    test_manifest = manifests["test"]
    fallback_epoch = terminal_epoch_for_run(epochs, smoke=smoke)
    for seed in train_seeds:
        for variant in variants:
            name = output_name(variant, seed, smoke, run_label)
            best_epoch = fallback_epoch if dry_run else _best_epoch_for(name, epochs, val_num_eval)[0]
            if best_epoch is None:
                print(f"SKIP selected test {name}: no validation-selected checkpoint")
                return False
            ok = eval_variant(
                variant,
                seed,
                split="test",
                manifest=test_manifest,
                epoch=best_epoch,
                dry_run=dry_run,
                force=force,
                smoke=smoke,
                num_samples=num_samples,
                n_steps=n_steps,
                solver_batch_size=solver_batch_size,
                test_chunk_size=test_chunk_size,
                run_label=run_label,
                retries=eval_retries,
            )
            if not ok:
                return False
    return True


def run_cycle(
    variants: list[Variant],
    train_seeds: list[int],
    epochs: list[int],
    *,
    manifests: dict[str, Path],
    dry_run: bool,
    force: bool,
    smoke: bool,
    wandb: bool,
    batch_size: int | None,
    val_manifest_kind: str,
    num_samples: int | None,
    n_steps: int | None,
    solver_batch_size: int | None = None,
    test_chunk_size: int | None = None,
    run_label: str = "",
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    log_every_n_steps: int | None = None,
    eval_retries: int = 0,
) -> bool:
    val_manifest = manifests["val_large" if val_manifest_kind == "large" else "val_small"]
    sorted_epochs = sorted(epochs)
    for seed in train_seeds:
        for variant in variants:
            name = output_name(variant, seed, smoke, run_label)
            for epoch_idx, epoch in enumerate(sorted_epochs):
                resume_training = weights_checkpoint_path(name).exists() or (
                    dry_run and epoch_idx > 0
                )
                ok = train_variant(
                    variant,
                    seed,
                    dry_run=dry_run,
                    force=force,
                    smoke=smoke,
                    wandb=wandb and not smoke,
                    batch_size=batch_size,
                    target_epoch=epoch,
                    resume=resume_training,
                    run_label=run_label,
                    limit_train_batches=limit_train_batches,
                    limit_val_batches=limit_val_batches,
                    log_every_n_steps=log_every_n_steps,
                )
                if not ok:
                    return False
                ok = eval_variant(
                    variant,
                    seed,
                    split="val",
                    manifest=val_manifest,
                    epoch=epoch,
                    dry_run=dry_run,
                    force=force,
                    smoke=smoke,
                    num_samples=num_samples,
                    n_steps=n_steps,
                    solver_batch_size=solver_batch_size,
                    run_label=run_label,
                    retries=eval_retries,
                )
                if not ok:
                    return False
    if not dry_run:
        ok = run_selected_test_evals(
            variants,
            train_seeds,
            epochs,
            manifests=manifests,
            dry_run=dry_run,
            force=force,
            smoke=smoke,
            num_samples=num_samples,
            n_steps=n_steps,
            solver_batch_size=solver_batch_size,
            test_chunk_size=test_chunk_size,
            run_label=run_label,
            val_manifest_kind=val_manifest_kind,
            eval_retries=eval_retries,
        )
        if not ok:
            return False
        write_report(
            variants,
            train_seeds,
            epochs,
            smoke=smoke,
            run_label=run_label,
            val_manifest_kind=val_manifest_kind,
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official-aligned PushT Stage 3.")
    parser.add_argument(
        "--mode",
        choices=["manifest", "train", "eval", "test", "report", "status", "cycle", "all"],
        default="status",
    )
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--train-seeds", type=int, nargs="*", default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--epochs", type=int, nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--log-every-n-steps", type=int, default=None)
    parser.add_argument("--eval-retries", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--solver-batch-size", type=int, default=None)
    parser.add_argument("--test-chunk-size", type=int, default=50)
    parser.add_argument("--val-manifest", choices=["small", "large"], default="small")
    args = parser.parse_args()

    variants = selected_variants(args.ids)
    epochs = parse_epochs(args.epochs, args.smoke)

    if args.mode == "status":
        status(
            variants,
            args.train_seeds,
            epochs,
            smoke=args.smoke,
            run_label=args.run_label,
            val_manifest_kind=args.val_manifest,
        )
        return
    if args.mode in ("manifest", "all", "eval", "test", "cycle"):
        manifests = ensure_manifests(force=args.force, smoke=args.smoke)
    else:
        manifests = {}
    if args.mode == "manifest":
        return

    ok = True
    if args.mode == "cycle":
        ok = run_cycle(
            variants,
            args.train_seeds,
            epochs,
            manifests=manifests,
            dry_run=args.dry_run,
            force=args.force,
            smoke=args.smoke,
            wandb=args.wandb,
            batch_size=args.batch_size,
            val_manifest_kind=args.val_manifest,
            num_samples=args.num_samples,
            n_steps=args.n_steps,
            solver_batch_size=args.solver_batch_size,
            test_chunk_size=args.test_chunk_size,
            run_label=args.run_label,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            log_every_n_steps=args.log_every_n_steps,
            eval_retries=args.eval_retries,
        )
        if not ok:
            sys.exit(1)
        return
    if args.mode in ("train", "all"):
        target_epoch = terminal_epoch_for_run(epochs, smoke=args.smoke)
        for seed in args.train_seeds:
            for variant in variants:
                ok = train_variant(
                    variant,
                    seed,
                    dry_run=args.dry_run,
                    force=args.force,
                    smoke=args.smoke,
                    wandb=args.wandb and not args.smoke,
                    batch_size=args.batch_size,
                    target_epoch=target_epoch,
                    run_label=args.run_label,
                    limit_train_batches=args.limit_train_batches,
                    limit_val_batches=args.limit_val_batches,
                    log_every_n_steps=args.log_every_n_steps,
                ) and ok
                if not ok:
                    sys.exit(1)
    if args.mode in ("eval", "all"):
        val_manifest = manifests["val_large" if args.val_manifest == "large" else "val_small"]
        for seed in args.train_seeds:
            for variant in variants:
                for epoch in epochs:
                    ok = eval_variant(
                        variant,
                        seed,
                        split="val",
                        manifest=val_manifest,
                        epoch=epoch,
                        dry_run=args.dry_run,
                        force=args.force,
                        smoke=args.smoke,
                        num_samples=args.num_samples,
                        n_steps=args.n_steps,
                        solver_batch_size=args.solver_batch_size,
                        run_label=args.run_label,
                        retries=args.eval_retries,
                    ) and ok
                    if not ok:
                        sys.exit(1)
    if args.mode in ("all",) and not args.dry_run:
        ok = run_selected_test_evals(
            variants,
            args.train_seeds,
            epochs,
            manifests=manifests,
            dry_run=args.dry_run,
            force=args.force,
            smoke=args.smoke,
            num_samples=args.num_samples,
            n_steps=args.n_steps,
            solver_batch_size=args.solver_batch_size,
            test_chunk_size=args.test_chunk_size,
            run_label=args.run_label,
            val_manifest_kind=args.val_manifest,
            eval_retries=args.eval_retries,
        ) and ok
        if not ok:
            sys.exit(1)
    if args.mode in ("test",):
        for seed in args.train_seeds:
            for variant in variants:
                for epoch in epochs:
                    ok = eval_variant(
                        variant,
                        seed,
                        split="test",
                        manifest=manifests["test"],
                        epoch=epoch,
                        dry_run=args.dry_run,
                        force=args.force,
                        smoke=args.smoke,
                        num_samples=args.num_samples,
                        n_steps=args.n_steps,
                        solver_batch_size=args.solver_batch_size,
                        test_chunk_size=args.test_chunk_size,
                        run_label=args.run_label,
                        retries=args.eval_retries,
                    ) and ok
                    if not ok:
                        sys.exit(1)
    if args.mode in ("report", "all"):
        if args.dry_run:
            print("DRY RUN: would write Stage 3 report")
        else:
            write_report(
                variants,
                args.train_seeds,
                epochs,
                smoke=args.smoke,
                run_label=args.run_label,
                val_manifest_kind=args.val_manifest,
            )


if __name__ == "__main__":
    main()
