from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
DIAGRAM_DIR = REPORT_DIR / "diagrams"
STABLEWM_HOME = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm")).expanduser()
RUNS_DIR = STABLEWM_HOME / "runs" / "pusht_expert_train"
PUSHT_DATASET_PATH = STABLEWM_HOME / "pusht_expert_train.h5"
ALLOW_STALE_FALLBACK = os.environ.get("AFLEWM_ALLOW_STALE_REPORT_FALLBACK") == "1"


EXPERIMENTS = {
    "Baseline LeWM": {
        "slug": "baseline",
        "run_name": "lewm_pusht_reliable",
        "params_fallback": 18_034_478,
        "color": "#2F4858",
        "results_files": ["pusht_results.txt", "pusht_results_seed43.txt"],
    },
    "AF-LeWM-lite v1": {
        "slug": "af_v1",
        "run_name": "aflewm_pusht_v1_reliable",
        "params_fallback": 18_167_150,
        "color": "#F28E2B",
        "results_files": ["pusht_results.txt", "pusht_results_seed43.txt"],
    },
    "AF-LeWM-lite v2": {
        "slug": "af_v2",
        "run_name": "aflewm_pusht_v2_reliable",
        "params_fallback": 18_236_792,
        "color": "#4E9F3D",
        "results_files": ["pusht_results.txt", "pusht_results_seed43.txt"],
    },
}


def run_git(args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_worktree_summary() -> dict[str, object]:
    status = run_git(["status", "--short"]) or ""
    entries = [line for line in status.splitlines() if line.strip()]
    return {
        "dirty": bool(entries),
        "status_entry_count": len(entries),
    }


def file_provenance(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text()) if path.exists() else {}


def load_epoch_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Lightning metrics file: {csv_path}")
    rows = list(csv.DictReader(csv_path.open(newline="")))
    return [
        row
        for row in rows
        if row.get("validate/loss_epoch") not in ("", None)
        or row.get("validate/pred_loss_epoch") not in ("", None)
        or row.get("validate/loss") not in ("", None)
        or row.get("validate/pred_loss") not in ("", None)
    ]


def row_float(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in ("", None):
            return float(value)
    return None


def extract_success_metrics(results_path: Path) -> dict[str, float]:
    if not results_path.exists():
        raise FileNotFoundError(f"Missing evaluation results file: {results_path}")
    text = results_path.read_text()
    eval_matches = re.findall(r"evaluation_time: ([0-9.]+) seconds", text)
    json_matches = re.findall(r"^metrics_json: (.+)$", text, re.M)
    if json_matches and eval_matches:
        metrics = json.loads(json_matches[-1])
        flags = metrics.get("episode_successes", [])
        successes = sum(bool(flag) for flag in flags)
        num_eval = len(flags)
        success_percent = 100.0 * successes / num_eval if num_eval else 0.0
        return {
            "success_percent": success_percent,
            "eval_time_seconds": float(eval_matches[-1]),
            "successes": successes,
            "num_eval": num_eval,
            "reported_success_percent": float(metrics["success_rate"]),
        }

    success_matches = re.findall(r"success_rate': ([0-9.]+)", text)
    episode_matches = re.findall(r"episode_successes': array\(\[(.*?)\]\), 'seeds'", text, re.S)
    if not success_matches or not eval_matches or not episode_matches:
        raise ValueError(f"Could not parse results file: {results_path}")

    flags = re.findall(r"True|False", episode_matches[-1])
    successes = sum(flag == "True" for flag in flags)
    num_eval = len(flags)
    success_percent = 100.0 * successes / num_eval if num_eval else 0.0
    return {
        "success_percent": success_percent,
        "eval_time_seconds": float(eval_matches[-1]),
        "successes": successes,
        "num_eval": num_eval,
        "reported_success_percent": float(success_matches[-1]),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0

    p_hat = successes / total
    denom = 1.0 + z * z / total
    center = (p_hat + z * z / (2.0 * total)) / denom
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total) / denom
    return center, spread


def load_existing_results() -> dict[str, object]:
    results_path = REPORT_DIR / "pusht_official_budget_results.json"
    if not results_path.exists():
        return {"summary": [], "curves": {}}
    return json.loads(results_path.read_text())


def build_summary() -> tuple[list[dict[str, object]], dict[str, list[dict[str, float]]], list[str]]:
    summary_rows: list[dict[str, object]] = []
    curves: dict[str, list[dict[str, float]]] = {}
    warnings: list[str] = []
    existing = load_existing_results()
    existing_summary = {row["model"]: row for row in existing.get("summary", [])}
    existing_curves = existing.get("curves", {})

    for name, meta in EXPERIMENTS.items():
        run_dir = RUNS_DIR / meta["run_name"]
        csv_path = run_dir / "metrics" / "metrics.csv"
        train_metadata_path = run_dir / "train_metadata.json"
        split_metadata_path = run_dir / "split_metadata.json"
        config_path = run_dir / "config.yaml"
        checkpoint_path = run_dir / f"{meta['run_name']}_epoch_10_object.ckpt"
        result_paths = [run_dir / filename for filename in meta["results_files"]]
        source_paths = [
            csv_path,
            train_metadata_path,
            split_metadata_path,
            config_path,
            checkpoint_path,
            *result_paths,
        ]

        if not all(path.exists() for path in source_paths):
            missing = [str(path) for path in source_paths if not path.exists()]
            if ALLOW_STALE_FALLBACK and name in existing_summary and name in existing_curves:
                warnings.append(f"{name}: using stale JSON fallback because sources are missing: {missing}")
                row = dict(existing_summary[name])
                row["source_mode"] = "stale_json_fallback"
                row["source_warnings"] = missing
                summary_rows.append(row)
                curves[name] = existing_curves[name]
                continue
            raise FileNotFoundError(f"Missing live sources for {name}: {missing}")

        rows = load_epoch_rows(csv_path)
        if not rows:
            raise ValueError(f"No validation epoch rows found in {csv_path}")

        train_metadata = read_json(train_metadata_path)
        split_metadata = read_json(split_metadata_path)
        seed_results = [extract_success_metrics(path) for path in result_paths]

        curve_points = []
        for row in rows:
            pred = row_float(row, "validate/pred_loss_epoch", "validate/pred_loss")
            sigreg = row_float(row, "validate/sigreg_loss_epoch", "validate/sigreg_loss") or 0.0
            if pred is None:
                continue
            curve_points.append(
                {
                    "epoch": int(float(row.get("epoch", len(curve_points)))) + 1,
                    "validate_pred_loss_epoch": pred,
                    "shared_core_loss_epoch": pred + 0.09 * sigreg,
                }
            )
        curves[name] = curve_points

        last = rows[-1]
        validate_pred_loss = row_float(last, "validate/pred_loss_epoch", "validate/pred_loss")
        validate_sigreg_loss = row_float(last, "validate/sigreg_loss_epoch", "validate/sigreg_loss") or 0.0
        validate_loss = row_float(last, "validate/loss_epoch", "validate/loss")
        if validate_pred_loss is None:
            raise ValueError(f"Missing validation prediction loss in {csv_path}")

        shared_core_loss = validate_pred_loss + 0.09 * validate_sigreg_loss
        aggregate_successes = sum(int(seed["successes"]) for seed in seed_results)
        aggregate_episodes = sum(int(seed["num_eval"]) for seed in seed_results)
        aggregate_success_rate = 100.0 * aggregate_successes / aggregate_episodes
        wilson_center, wilson_spread = wilson_interval(aggregate_successes, aggregate_episodes)
        summary_rows.append(
            {
                "model": name,
                "slug": meta["slug"],
                "run_name": meta["run_name"],
                "source_mode": "live",
                "params": int(train_metadata.get("model_params", meta["params_fallback"])),
                "train_elapsed_seconds": train_metadata.get("train_elapsed_seconds"),
                "evaluation_time_seed42_seconds": seed_results[0]["eval_time_seconds"],
                "evaluation_time_seed43_seconds": seed_results[1]["eval_time_seconds"],
                "evaluation_time_mean_seconds": mean(seed["eval_time_seconds"] for seed in seed_results),
                "success_seed42_count": seed_results[0]["successes"],
                "success_seed42_total": seed_results[0]["num_eval"],
                "success_seed43_count": seed_results[1]["successes"],
                "success_seed43_total": seed_results[1]["num_eval"],
                "success_seed42_percent": seed_results[0]["success_percent"],
                "success_seed43_percent": seed_results[1]["success_percent"],
                "aggregate_successes": aggregate_successes,
                "aggregate_episodes": aggregate_episodes,
                "aggregate_success_percent": aggregate_success_rate,
                "aggregate_wilson95_low_percent": 100.0 * (wilson_center - wilson_spread),
                "aggregate_wilson95_high_percent": 100.0 * (wilson_center + wilson_spread),
                "validate_loss_epoch": validate_loss,
                "validate_pred_loss_epoch": validate_pred_loss,
                "validate_sigreg_loss_epoch": validate_sigreg_loss,
                "shared_core_loss_epoch": shared_core_loss,
                "validate_appearance_inv_loss_epoch": row_float(last, "validate/appearance_inv_loss_epoch"),
                "validate_appearance_indep_loss_epoch": row_float(last, "validate/appearance_indep_loss_epoch"),
                "validate_appearance_nuisance_loss_epoch": row_float(last, "validate/appearance_nuisance_loss_epoch"),
                "validate_dynamics_nuisance_loss_epoch": row_float(last, "validate/dynamics_nuisance_loss_epoch"),
                "train_episode_count": split_metadata.get("train_episode_count"),
                "val_episode_count": split_metadata.get("val_episode_count"),
                "train_sample_count": split_metadata.get("train_sample_count"),
                "val_sample_count": split_metadata.get("val_sample_count"),
                "source_files": [file_provenance(path) for path in source_paths],
            }
        )

    return summary_rows, curves, warnings


def write_json(summary_rows: list[dict[str, object]], curves: dict[str, list[dict[str, float]]], warnings: list[str]) -> None:
    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": "live" if not warnings else "mixed",
        "git_head_at_generation": run_git(["rev-parse", "HEAD"]),
        "git_worktree_at_generation": git_worktree_summary(),
        "artifact_commit_note": (
            "This file is generated before the commit that may contain it; "
            "use git_head_at_generation for the source tree used during generation."
        ),
        "stablewm_home": str(STABLEWM_HOME),
        "dataset_source": file_provenance(PUSHT_DATASET_PATH),
        "warnings": warnings,
        "summary": summary_rows,
        "curves": curves,
    }
    out_path = REPORT_DIR / "pusht_official_budget_results.json"
    out_path.write_text(json.dumps(payload, indent=2))


def write_csv(summary_rows: list[dict[str, object]]) -> None:
    out_path = REPORT_DIR / "pusht_official_budget_summary.csv"
    fieldnames = [
        "model",
        "run_name",
        "source_mode",
        "params",
        "train_elapsed_seconds",
        "success_seed42_count",
        "success_seed42_total",
        "success_seed43_count",
        "success_seed43_total",
        "success_seed42_percent",
        "success_seed43_percent",
        "aggregate_successes",
        "aggregate_episodes",
        "aggregate_success_percent",
        "aggregate_wilson95_low_percent",
        "aggregate_wilson95_high_percent",
        "validate_loss_epoch",
        "validate_pred_loss_epoch",
        "validate_sigreg_loss_epoch",
        "shared_core_loss_epoch",
        "train_episode_count",
        "val_episode_count",
        "train_sample_count",
        "val_sample_count",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_success(summary_rows: list[dict[str, object]]) -> None:
    labels = [row["model"] for row in summary_rows]
    values = [row["aggregate_success_percent"] for row in summary_rows]
    lower = [row["aggregate_wilson95_low_percent"] for row in summary_rows]
    upper = [row["aggregate_wilson95_high_percent"] for row in summary_rows]
    yerr = [
        [value - lo for value, lo in zip(values, lower)],
        [hi - value for value, hi in zip(values, upper)],
    ]
    colors = [EXPERIMENTS[row["model"]]["color"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.58, yerr=yerr, capsize=7, ecolor="#374151")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, max(upper) + 2.5)
    ax.set_title("PushT Official: Two-Seed Aggregate Planning Success")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, row, value in zip(bars, summary_rows, values):
        label = f"{row['aggregate_successes']}/{row['aggregate_episodes']}"
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, label, ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "pusht_success_rate.png", dpi=220)
    plt.close(fig)


def plot_core_loss(curves: dict[str, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for name, points in curves.items():
        ax.plot(
            [point["epoch"] for point in points],
            [point["shared_core_loss_epoch"] for point in points],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=EXPERIMENTS[name]["color"],
            label=name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Shared Core Loss")
    ax.set_title("PushT Official: Shared JEPA Core Loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "pusht_val_core_loss.png", dpi=220)
    plt.close(fig)


def add_box(ax, xy, text, color, width=2.5, height=0.78):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.06",
        linewidth=1.15,
        edgecolor="#1F2937",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8.5, color="#111827")
    return (x, y, width, height)


def add_arrow(ax, src, dst, label=None):
    sx, sy, sw, sh = src
    dx, dy, dw, dh = dst
    start = (sx + sw, sy + sh / 2)
    end = (dx, dy + dh / 2)
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#374151")
    ax.add_patch(arrow)
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label, fontsize=7.4, ha="center", color="#4B5563")


def save_diagram_sources() -> None:
    DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "aflewm_model_pipeline.mmd": """flowchart LR
  A["Official PushT sequence<br/>pixels / action / proprio / state"] --> B["Image preprocessing<br/>ImageNet norm + 224 resize"]
  B --> C["ViT-tiny encoder<br/>CLS features"]
  C --> D["Dynamics projector<br/>emb"]
  A --> E["Action block encoder<br/>5 raw actions -> one block"]
  D --> F["AR predictor<br/>3 history latents + action blocks"]
  E --> F
  F --> G["Predicted future emb"]
  G --> H["CEM planner<br/>terminal latent goal cost"]
  H --> I["Env action<br/>inverse normalize + clip"]
  C --> J["Appearance projector<br/>app_emb"]
  J --> K["AF v1 shaping losses<br/>invariance + independence"]
  J --> L["AF v2 nuisance losses<br/>appearance head + GRL head"]
  K --> M["Training objective only"]
  L --> M
""",
        "pusht_experiment_flow.mmd": """flowchart LR
  A["Official PushT dataset"] --> B["Validate dataset and environment"]
  B --> C["Episode-disjoint split"]
  C --> D["Train-only action/proprio/state normalizers"]
  D --> E["Train three matched configs<br/>LeWM / AF v1 / AF v2"]
  E --> F["Epoch-10 object checkpoints<br/>fresh-run fail-fast"]
  F --> G["Eval seed 42<br/>50 sampled starts"]
  F --> H["Eval seed 43<br/>50 sampled starts"]
  G --> I["Aggregate 100 trials<br/>Wilson 95 percent intervals"]
  H --> I
  I --> J["JSON / CSV / plots / PDF report"]
""",
        "implementation_fix_map.mmd": """flowchart LR
  A["Prior risk: clip-level split leakage"] --> B["Episode-disjoint train/val split"]
  C["Prior risk: full-data normalizer"] --> D["Normalizer fitted on train episodes"]
  E["Prior risk: stale checkpoint resume"] --> F["Fresh-run fail-fast unless resume=True"]
  G["Prior risk: invalid CEM candidates"] --> H["Bound normalized candidates and elites"]
  I["Prior risk: weak result provenance"] --> J["Store split/train/eval source metadata"]
""",
    }
    for filename, text in sources.items():
        (DIAGRAM_DIR / filename).write_text(text)


def draw_model_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(14.8, 5.8))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.set_title("AF-LeWM-lite Model Pipeline", fontsize=15, fontweight="bold", pad=12)

    boxes = {}
    boxes["data"] = add_box(ax, (0.2, 2.65), "Official PushT\npixels / action / state", "#E5E7EB", width=1.85)
    boxes["pre"] = add_box(ax, (2.35, 2.65), "Image preprocess\n224 + ImageNet norm", "#DBEAFE", width=1.85)
    boxes["vit"] = add_box(ax, (4.5, 2.65), "ViT-tiny encoder\nCLS features", "#DBEAFE", width=1.85)
    boxes["dyn"] = add_box(ax, (6.65, 2.65), "Dynamics projector\nemb", "#D1FAE5", width=1.85)
    boxes["pred"] = add_box(ax, (8.8, 2.65), "AR predictor\n3-frame history", "#D1FAE5", width=1.85)
    boxes["cem"] = add_box(ax, (10.95, 2.65), "CEM planner\nterminal goal cost", "#FCE7F3", width=1.85)
    boxes["env"] = add_box(ax, (13.1, 2.65), "Env action\ninverse norm + clip", "#F3F4F6", width=1.5)
    boxes["act"] = add_box(ax, (6.65, 1.35), "Action block encoder\n5 actions per block", "#FEF3C7", width=1.85)
    boxes["app"] = add_box(ax, (6.65, 4.0), "Appearance projector\napp_emb", "#FFE4E6", width=1.85)
    boxes["loss"] = add_box(ax, (8.8, 4.0), "AF shaping losses\nv1 + v2", "#FFE4E6", width=1.85)
    boxes["objective"] = add_box(ax, (10.95, 4.0), "Training objective\ncore + AF losses", "#F3F4F6", width=1.85)

    add_arrow(ax, boxes["data"], boxes["pre"], "pixels")
    add_arrow(ax, boxes["pre"], boxes["vit"])
    add_arrow(ax, boxes["vit"], boxes["dyn"])
    add_arrow(ax, boxes["dyn"], boxes["pred"])
    add_arrow(ax, boxes["data"], boxes["act"], "actions")
    add_arrow(ax, boxes["act"], boxes["pred"])
    add_arrow(ax, boxes["pred"], boxes["cem"])
    add_arrow(ax, boxes["cem"], boxes["env"])
    add_arrow(ax, boxes["vit"], boxes["app"])
    add_arrow(ax, boxes["app"], boxes["loss"])
    add_arrow(ax, boxes["loss"], boxes["objective"])
    ax.text(8.05, 0.55, "Planning uses dynamics emb only; appearance heads shape training.", ha="center", fontsize=9.5, color="#374151")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "aflewm_model_pipeline.png", dpi=220)
    plt.close(fig)


def draw_experiment_flow() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.set_title("Reliable PushT Experiment Flow", fontsize=15, fontweight="bold", pad=12)

    boxes = {}
    xs = [0.2, 2.35, 4.5, 6.65, 8.8, 10.95]
    labels = [
        "Official PushT\nHDF5 dataset",
        "Episode split\ntrain / val disjoint",
        "Train-only\nnormalizers",
        "Train 3 configs\nmatched budget",
        "Epoch-10\nobject ckpts",
        "Report assets\nJSON / CSV / PDF",
    ]
    for idx, (x, label) in enumerate(zip(xs, labels)):
        boxes[idx] = add_box(ax, (x, 3.1), label, "#E0F2FE", width=1.85, height=0.82)
    for idx in range(5):
        add_arrow(ax, boxes[idx], boxes[idx + 1])

    eval42 = add_box(ax, (7.1, 1.45), "Eval seed 42\n50 starts", "#ECFCCB", width=2.0)
    eval43 = add_box(ax, (9.45, 1.45), "Eval seed 43\n50 starts", "#ECFCCB", width=2.0)
    agg = add_box(ax, (11.75, 1.45), "Aggregate\nWilson CI", "#ECFCCB", width=1.25)
    add_arrow(ax, boxes[4], eval42)
    add_arrow(ax, boxes[4], eval43)
    add_arrow(ax, eval42, agg)
    add_arrow(ax, eval43, agg)
    ax.text(6.55, 0.65, "Fresh-run fail-fast prevents stale checkpoint continuation.", fontsize=9.5, color="#374151")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "pusht_experiment_flow.png", dpi=220)
    plt.close(fig)


def draw_fix_map() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    ax.set_title("Implementation Fix Map", fontsize=15, fontweight="bold", pad=12)

    left = [
        "Clip-level\nsplit leakage",
        "Full-data\nnormalizer",
        "Stale checkpoint\nresume",
        "Unbounded\nCEM candidates",
        "Weak provenance\nfor report",
    ]
    right = [
        "Episode-disjoint\nsplit",
        "Train-episode\nnormalizer",
        "Fresh-run\nfail-fast",
        "Clamp normalized\ncandidate blocks",
        "Split / train / eval\nmetadata",
    ]
    for i, (ltext, rtext) in enumerate(zip(left, right)):
        y = 3.75 - i * 0.72
        lbox = add_box(ax, (0.35, y), ltext, "#FEE2E2", width=3.0, height=0.52)
        rbox = add_box(ax, (8.85, y), rtext, "#DCFCE7", width=3.0, height=0.52)
        add_arrow(ax, lbox, rbox, "fix")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "implementation_fix_map.png", dpi=220)
    plt.close(fig)


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_num(value: object, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def fmt_min(seconds: object) -> str:
    if seconds is None:
        return "--"
    return f"{float(seconds) / 60.0:.2f}"


def write_tex(summary_rows: list[dict[str, object]]) -> None:
    report_date = datetime.now().strftime("%B %d, %Y")
    best = max(summary_rows, key=lambda row: float(row["aggregate_success_percent"]))
    total_trials = int(best["aggregate_episodes"])
    result_lines = []
    uncertainty_lines = []
    for row in summary_rows:
        result_lines.append(
            " & ".join(
                [
                    tex_escape(row["model"]),
                    f"{int(row['params']) / 1e6:.2f}",
                    fmt_min(row["train_elapsed_seconds"]),
                    fmt_num(row["shared_core_loss_epoch"], 5),
                    f"{int(row['success_seed42_count'])}/{int(row['success_seed42_total'])}",
                    f"{int(row['success_seed43_count'])}/{int(row['success_seed43_total'])}",
                    f"{int(row['aggregate_successes'])}/{int(row['aggregate_episodes'])} = {float(row['aggregate_success_percent']):.1f}\\%",
                ]
            )
            + r" \\"
        )
        uncertainty_lines.append(
            " & ".join(
                [
                    tex_escape(row["model"]),
                    f"{int(row['aggregate_successes'])}/{int(row['aggregate_episodes'])} = {float(row['aggregate_success_percent']):.1f}\\%",
                    f"[{float(row['aggregate_wilson95_low_percent']):.2f}\\%, {float(row['aggregate_wilson95_high_percent']):.2f}\\%]",
                ]
            )
            + r" \\"
        )

    tex = rf"""\documentclass[11pt]{{article}}

\usepackage[margin=0.78in]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{microtype}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{subcaption}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}

\setmainfont{{Georgia}}
\setsansfont{{Segoe UI}}
\setmonofont{{Consolas}}

\definecolor{{ink}}{{HTML}}{{1F2937}}
\definecolor{{accent}}{{HTML}}{{0F766E}}
\definecolor{{accentlight}}{{HTML}}{{E6FFFB}}
\definecolor{{muted}}{{HTML}}{{5B6470}}

\hypersetup{{
  colorlinks=true,
  linkcolor=accent,
  urlcolor=accent
}}

\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.52em}}

\begin{{document}}
\color{{ink}}

{{\sffamily
\begin{{center}}
  {{\Large\bfseries AF-LeWM-lite on Official PushT}}\\[0.35em]
  {{\large Reliable Rerun Summary}}\\[0.45em]
  {{\small {report_date}}}
\end{{center}}
}}

\vspace{{0.2em}}
\fcolorbox{{accent}}{{accentlight}}{{%
  \parbox{{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}}{{
    \textbf{{Bottom line.}} The reliable rerun reports \textbf{{{int(best['aggregate_successes'])}/{total_trials}}} successes for \textbf{{{tex_escape(best['model'])}}}. All models remain weak under the short matched budget, and the Wilson intervals overlap, so the current evidence supports an exploratory ranking rather than a benchmark-level claim.
  }}
}}

\section*{{Architecture and Design Principle}}
AF-LeWM-lite keeps the LeWM planning contract intact. Pixels are encoded by a shared ViT-tiny backbone; the dynamics projection head produces \texttt{{emb}}, action blocks are encoded separately, and the autoregressive predictor rolls the dynamics latent forward under candidate actions. Planning uses terminal latent distance between predicted dynamics and the goal dynamics latent.

The AF extension adds an appearance projection head \texttt{{app\_emb}} used only for representation shaping during training. v1 adds dynamics invariance under appearance augmentation and a cross-covariance penalty between \texttt{{emb}} and \texttt{{app\_emb}}. v2 adds sequence-consistent augmentation parameters, stop-gradient invariance, an appearance nuisance head, and a gradient-reversal nuisance head on the dynamics branch.

\begin{{figure}}[h]
\centering
\includegraphics[width=\linewidth]{{aflewm_model_pipeline.png}}
\caption{{Model pipeline. The planner sees only the dynamics branch; appearance heads affect training losses.}}
\end{{figure}}

\section*{{Reliable Experiment Protocol}}
The rerun uses the official \texttt{{pusht\_expert\_train.h5}} dataset, an episode-disjoint 90/10 train/validation split, normalizers fitted only on training episodes, and fresh-run guards that refuse to mix stale checkpoint or metrics artifacts with a new run. Each model is trained for 10 epochs with 200 training batches per epoch, 20 validation batches per epoch, batch size 4, bf16 precision, and seed 3072.

Each epoch-10 object checkpoint is evaluated on two independent 50-start samples, \texttt{{seed=42}} and \texttt{{seed=43}}. The report aggregates 100 trials per model and reports Wilson 95\% intervals. Cross-model loss comparisons use the shared JEPA core objective \(\mathcal{{L}}_{{core}} = \mathcal{{L}}_{{pred}} + 0.09\mathcal{{L}}_{{sigreg}}\).

\begin{{figure}}[h]
\centering
\includegraphics[width=\linewidth]{{pusht_experiment_flow.png}}
\caption{{Experiment flow from official dataset to two-seed aggregate report.}}
\end{{figure}}

\section*{{Implementation Fixes}}
\begin{{itemize}}[leftmargin=1.2em, itemsep=0.16em, topsep=0.16em]
  \item Training now uses episode-disjoint subsets rather than overlapping clip-level random split.
  \item Action, proprioception, and state normalizers are fitted on training episodes only.
  \item Fresh reliable runs fail fast when old metrics or checkpoints already exist.
  \item Evaluation reconstructs the raw 15-step history window and compresses it to the model's 3-frame context.
  \item Goal indexing is asserted against the planned raw-step horizon.
  \item CEM candidate actions, elite means, and returned normalized plans are clamped to normalized env action bounds before inverse transform.
  \item Evaluation writes row, episode, and start-step provenance for every sampled rollout.
\end{{itemize}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.95\linewidth]{{implementation_fix_map.png}}
\caption{{Reliability fixes that directly affect whether the experiment tests the intended design.}}
\end{{figure}}

\section*{{Results}}
\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
Model & Params (M) & Train (min) & Core loss & Seed 42 & Seed 43 & Aggregate \\
\midrule
{chr(10).join(result_lines)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{lrr}}
\toprule
Model & Aggregate success & Wilson 95\% CI \\
\midrule
{chr(10).join(uncertainty_lines)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[h]
\centering
\begin{{subfigure}}[t]{{0.48\linewidth}}
  \centering
  \includegraphics[width=\linewidth]{{pusht_success_rate.png}}
  \caption{{Two-seed planning success with Wilson intervals.}}
\end{{subfigure}}
\hfill
\begin{{subfigure}}[t]{{0.48\linewidth}}
  \centering
  \includegraphics[width=\linewidth]{{pusht_val_core_loss.png}}
  \caption{{Shared JEPA core loss across training epochs.}}
\end{{subfigure}}
\end{{figure}}

\section*{{Interpretation}}
The corrected result should be read conservatively. The best model in this rerun is \textbf{{{tex_escape(best['model'])}}}, but the aggregate success counts are small and the intervals overlap. The current study is useful as a reliability-checked ablation of the AF shaping idea; it does not establish a strong PushT control result.

The most important conclusion is methodological: after removing split leakage, full-data normalization, stale checkpoint continuation, and invalid CEM candidate actions, any remaining weakness is visible in the audited protocol rather than hidden in the implementation.

\vfill
{{\small\color{{muted}}
Generated from \texttt{{tools/generate\_pusht\_official\_report\_assets.py}} using live run metrics, split metadata, train metadata, and evaluation result files under \texttt{{STABLEWM\_HOME}}.
}}

\end{{document}}
"""
    (REPORT_DIR / "pusht_aflewm_official_summary.tex").write_text(tex)


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    save_diagram_sources()
    draw_model_pipeline()
    draw_experiment_flow()
    draw_fix_map()
    summary_rows, curves, warnings = build_summary()
    write_json(summary_rows, curves, warnings)
    write_csv(summary_rows)
    plot_success(summary_rows)
    plot_core_loss(curves)
    write_tex(summary_rows)
    print(f"Wrote assets to {REPORT_DIR}")


if __name__ == "__main__":
    main()
