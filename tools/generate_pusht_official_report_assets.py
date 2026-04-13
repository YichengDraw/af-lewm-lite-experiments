from __future__ import annotations

import csv
import json
import math
import re
from statistics import mean
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
RUNS_DIR = Path.home() / ".stable-wm" / "runs" / "pusht_expert_train"


EXPERIMENTS = {
    "Baseline LeWM": {
        "slug": "baseline",
        "log_version": 1,
        "results_dir": RUNS_DIR / "lewm_pusht_implfix_budget",
        "train_elapsed_seconds": 477.70,
        "params": 18_034_478,
        "color": "#2F4858",
        "results_files": ["pusht_results.txt", "pusht_results_seed43.txt"],
    },
    "AF-LeWM-lite v1": {
        "slug": "af_v1",
        "log_version": 3,
        "results_dir": RUNS_DIR / "aflewm_pusht_implfix_budget_rerun",
        "train_elapsed_seconds": 900.30,
        "params": 18_167_150,
        "color": "#F28E2B",
        "results_files": ["pusht_results.txt", "pusht_results_seed43.txt"],
    },
    "AF-LeWM-lite v2": {
        "slug": "af_v2",
        "log_version": 4,
        "results_dir": RUNS_DIR / "aflewm_pusht_v2_implfix_budget_rerun",
        "train_elapsed_seconds": 987.90,
        "params": 18_236_792,
        "color": "#4E9F3D",
        "results_files": ["pusht_results.txt", "pusht_results_seed43.txt"],
    },
}


def load_epoch_rows(csv_path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(csv_path.open(newline="")))
    return [row for row in rows if row.get("validate/loss_epoch") not in ("", None)]


def extract_success_metrics(results_path: Path) -> dict[str, float]:
    text = results_path.read_text()
    success_matches = re.findall(r"success_rate': ([0-9.]+)", text)
    eval_matches = re.findall(r"evaluation_time: ([0-9.]+) seconds", text)
    episode_match = re.search(r"episode_successes': array\(\[(.*?)\]\), 'seeds'", text, re.S)
    if not success_matches or not eval_matches or episode_match is None:
        raise ValueError(f"Could not parse results file: {results_path}")

    flags = re.findall(r"True|False", episode_match.group(1))
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


def build_summary() -> tuple[list[dict[str, object]], dict[str, list[dict[str, float]]]]:
    summary_rows: list[dict[str, object]] = []
    curves: dict[str, list[dict[str, float]]] = {}

    for name, meta in EXPERIMENTS.items():
        csv_path = ROOT / "lightning_logs" / f"version_{meta['log_version']}" / "metrics.csv"
        rows = load_epoch_rows(csv_path)
        seed_results = [extract_success_metrics(meta["results_dir"] / filename) for filename in meta["results_files"]]

        curves[name] = [
            {
                "epoch": int(row["epoch"]) + 1,
                "validate_pred_loss_epoch": float(row["validate/pred_loss_epoch"]),
                "shared_core_loss_epoch": float(row["validate/pred_loss_epoch"])
                + 0.09 * float(row["validate/sigreg_loss_epoch"]),
            }
            for row in rows
        ]

        last = rows[-1]
        validate_pred_loss = float(last["validate/pred_loss_epoch"])
        validate_sigreg_loss = float(last["validate/sigreg_loss_epoch"])
        shared_core_loss = validate_pred_loss + 0.09 * validate_sigreg_loss
        aggregate_successes = sum(int(seed["successes"]) for seed in seed_results)
        aggregate_episodes = sum(int(seed["num_eval"]) for seed in seed_results)
        aggregate_success_rate = 100.0 * aggregate_successes / aggregate_episodes
        wilson_center, wilson_spread = wilson_interval(aggregate_successes, aggregate_episodes)
        summary_rows.append(
            {
                "model": name,
                "slug": meta["slug"],
                "params": meta["params"],
                "train_elapsed_seconds": meta["train_elapsed_seconds"],
                "evaluation_time_seed42_seconds": seed_results[0]["eval_time_seconds"],
                "evaluation_time_seed43_seconds": seed_results[1]["eval_time_seconds"],
                "evaluation_time_mean_seconds": mean(seed["eval_time_seconds"] for seed in seed_results),
                "success_seed42_percent": seed_results[0]["success_percent"],
                "success_seed43_percent": seed_results[1]["success_percent"],
                "aggregate_successes": aggregate_successes,
                "aggregate_episodes": aggregate_episodes,
                "aggregate_success_percent": aggregate_success_rate,
                "aggregate_wilson95_low_percent": 100.0 * (wilson_center - wilson_spread),
                "aggregate_wilson95_high_percent": 100.0 * (wilson_center + wilson_spread),
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
    ax.set_title("PushT Official: Shared Core Loss Across Epochs")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "pusht_val_core_loss.png", dpi=220)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    summary_rows, curves = build_summary()
    write_json(summary_rows, curves)
    write_csv(summary_rows)
    plot_success(summary_rows)
    plot_core_loss(curves)
    print(f"Wrote assets to {REPORT_DIR}")


if __name__ == "__main__":
    main()
