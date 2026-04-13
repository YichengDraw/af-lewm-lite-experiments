from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
RUNS_DIR = Path.home() / ".stable-wm" / "runs" / "pusht_expert_train"


EXPERIMENTS = {
    "Baseline LeWM": {
        "slug": "baseline",
        "log_version": 21,
        "results_dir": RUNS_DIR / "lewm_pusht_official_budget",
        "train_elapsed_seconds": 466.52,
        "params": 18_034_478,
        "color": "#2F4858",
    },
    "AF-LeWM-lite v1": {
        "slug": "af_v1",
        "log_version": 22,
        "results_dir": RUNS_DIR / "aflewm_pusht_official_budget",
        "train_elapsed_seconds": 750.55,
        "params": 18_167_150,
        "color": "#F28E2B",
    },
    "AF-LeWM-lite v2": {
        "slug": "af_v2",
        "log_version": 23,
        "results_dir": RUNS_DIR / "aflewm_pusht_v2_official_budget",
        "train_elapsed_seconds": 805.34,
        "params": 18_236_792,
        "color": "#4E9F3D",
    },
}


def load_epoch_rows(csv_path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(csv_path.open(newline="")))
    return [row for row in rows if row.get("validate/loss_epoch") not in ("", None)]


def extract_success_metrics(results_path: Path) -> tuple[float, float]:
    text = results_path.read_text()
    success = float(re.search(r"success_rate': ([0-9.]+)", text).group(1))
    eval_time = float(re.search(r"evaluation_time: ([0-9.]+) seconds", text).group(1))
    return success, eval_time


def build_summary() -> tuple[list[dict[str, object]], dict[str, list[dict[str, float]]]]:
    summary_rows: list[dict[str, object]] = []
    curves: dict[str, list[dict[str, float]]] = {}

    for name, meta in EXPERIMENTS.items():
        csv_path = ROOT / "lightning_logs" / f"version_{meta['log_version']}" / "metrics.csv"
        rows = load_epoch_rows(csv_path)
        success_rate, eval_time = extract_success_metrics(meta["results_dir"] / "pusht_results.txt")

        curves[name] = [
            {
                "epoch": int(row["epoch"]) + 1,
                "validate_pred_loss_epoch": float(row["validate/pred_loss_epoch"]),
            }
            for row in rows
        ]

        last = rows[-1]
        validate_pred_loss = float(last["validate/pred_loss_epoch"])
        validate_sigreg_loss = float(last["validate/sigreg_loss_epoch"])
        shared_core_loss = validate_pred_loss + 0.09 * validate_sigreg_loss
        summary_rows.append(
            {
                "model": name,
                "slug": meta["slug"],
                "params": meta["params"],
                "train_elapsed_seconds": meta["train_elapsed_seconds"],
                "evaluation_time_seconds": eval_time,
                "success_rate": success_rate,
                "validate_loss_epoch": float(last["validate/loss_epoch"]),
                "validate_pred_loss_epoch": validate_pred_loss,
                "validate_sigreg_loss_epoch": validate_sigreg_loss,
                "shared_core_loss_epoch": shared_core_loss,
                "validate_appearance_inv_loss_epoch": (
                    float(last["validate/appearance_inv_loss_epoch"])
                    if last.get("validate/appearance_inv_loss_epoch")
                    else None
                ),
                "validate_appearance_indep_loss_epoch": (
                    float(last["validate/appearance_indep_loss_epoch"])
                    if last.get("validate/appearance_indep_loss_epoch")
                    else None
                ),
                "validate_appearance_nuisance_loss_epoch": (
                    float(last["validate/appearance_nuisance_loss_epoch"])
                    if last.get("validate/appearance_nuisance_loss_epoch")
                    else None
                ),
                "validate_dynamics_nuisance_loss_epoch": (
                    float(last["validate/dynamics_nuisance_loss_epoch"])
                    if last.get("validate/dynamics_nuisance_loss_epoch")
                    else None
                ),
            }
        )

    return summary_rows, curves


def write_json(summary_rows: list[dict[str, object]], curves: dict[str, list[dict[str, float]]]) -> None:
    payload = {"summary": summary_rows, "curves": curves}
    out_path = REPORT_DIR / "pusht_official_budget_results.json"
    out_path.write_text(json.dumps(payload, indent=2))


def write_csv(summary_rows: list[dict[str, object]]) -> None:
    out_path = REPORT_DIR / "pusht_official_budget_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_success(summary_rows: list[dict[str, object]]) -> None:
    labels = [row["model"] for row in summary_rows]
    values = [row["success_rate"] for row in summary_rows]
    colors = [EXPERIMENTS[row["model"]]["color"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, max(values) + 3)
    ax.set_title("PushT Official: 50-Episode Planning Success")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{value:.1f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "pusht_success_rate.png", dpi=220)
    plt.close(fig)


def plot_pred_loss(curves: dict[str, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for name, points in curves.items():
        ax.plot(
            [point["epoch"] for point in points],
            [point["validate_pred_loss_epoch"] for point in points],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=EXPERIMENTS[name]["color"],
            label=name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Pred Loss")
    ax.set_title("PushT Official: Validation Prediction Loss Across Epochs")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "pusht_val_pred_loss.png", dpi=220)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    summary_rows, curves = build_summary()
    write_json(summary_rows, curves)
    write_csv(summary_rows)
    plot_success(summary_rows)
    plot_pred_loss(curves)
    print(f"Wrote assets to {REPORT_DIR}")


if __name__ == "__main__":
    main()
