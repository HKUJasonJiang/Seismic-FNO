"""Plot O3 sampling-resolution OOD results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODELS = ["FNO-Large", "Transformer", "BiLSTM"]
RUNS = ["S-20Hz", "S-25Hz", "S-50Hz"]
COLORS = {"FNO-Large": "#1f77b4", "Transformer": "#d62728", "BiLSTM": "#2ca02c"}
MARKERS = {"S-20Hz": "o", "S-25Hz": "s", "S-50Hz": "^"}


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_overall(run_root: Path) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for run in RUNS:
            path = run_root / model / run / "evaluation_overall.csv"
            row = pd.read_csv(path).iloc[0].to_dict()
            row["train_hz"] = int(run.split("-")[1].replace("Hz", ""))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_overall(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
    for model in MODELS:
        sub = df[df["model"] == model].sort_values("train_hz")
        axes[0].plot(sub["train_hz"], sub["mse"], color=COLORS[model], marker="o", lw=2.2, label=model)
        axes[1].plot(sub["train_hz"], sub["r2"], color=COLORS[model], marker="o", lw=2.2, label=model)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training sampling rate [Hz]")
    axes[0].set_ylabel("Fixed 50Hz test MSE")
    axes[0].set_title("Sampling OOD: fixed-test MSE")
    axes[1].set_xlabel("Training sampling rate [Hz]")
    axes[1].set_ylabel("Fixed 50Hz test R2")
    axes[1].set_title("Sampling OOD: fixed-test R2")
    for ax in axes:
        ax.set_xticks([20, 25, 50])
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    savefig(fig, output_dir / "sampling_ood_overall_mse_r2")


def plot_cost(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    for ax, (metric, title) in zip(
        axes,
        [
            ("mse", "MSE"),
            ("train_time_s", "Training time [s]"),
            ("peak_gpu_gb", "Peak GPU [GB]"),
        ],
    ):
        labels = []
        values = []
        colors = []
        for model in MODELS:
            for run in RUNS:
                row = df[(df["model"] == model) & (df["run_id"] == run)].iloc[0]
                labels.append(f"{model}\n{run}")
                values.append(row[metric])
                colors.append(COLORS[model])
        ax.bar(labels, values, color=colors, width=0.75)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=65, labelsize=7)
        if metric == "mse":
            ax.set_yscale("log")
    savefig(fig, output_dir / "sampling_ood_accuracy_cost_bars")


def plot_training_curves(run_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 10.2), constrained_layout=True)
    for row_idx, model in enumerate(MODELS):
        for run in RUNS:
            df = pd.read_csv(run_root / model / run / "training_log.csv")
            axes[row_idx, 0].plot(df["epoch"], df["val_mse"], color=COLORS[model], marker=MARKERS[run], markevery=10, lw=1.7, label=run)
            axes[row_idx, 1].plot(df["epoch"], df["val_r2"], color=COLORS[model], marker=MARKERS[run], markevery=10, lw=1.7, label=run)
        axes[row_idx, 0].set_yscale("log")
        axes[row_idx, 0].set_ylabel(f"{model}\nval MSE")
        axes[row_idx, 1].set_ylabel(f"{model}\nval R2")
        for ax in axes[row_idx]:
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
    axes[0, 0].set_title("Validation MSE by training sampling rate")
    axes[0, 1].set_title("Validation R2 by training sampling rate")
    savefig(fig, output_dir / "sampling_ood_training_curves")


def plot_by_scale(run_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.2), constrained_layout=True)
    for row_idx, model in enumerate(MODELS):
        for run in RUNS:
            df = pd.read_csv(run_root / model / run / "evaluation_by_scale.csv")
            axes[row_idx, 0].plot(df["median_pga_ms2"], df["mse"], lw=1.8, marker=MARKERS[run], ms=3, label=run)
            axes[row_idx, 1].plot(df["median_pga_ms2"], df["r2"], lw=1.8, marker=MARKERS[run], ms=3, label=run)
        axes[row_idx, 0].set_yscale("log")
        axes[row_idx, 0].set_ylabel(f"{model}\nMSE")
        axes[row_idx, 1].set_ylabel(f"{model}\nR2")
        for ax in axes[row_idx]:
            ax.set_xlabel("Median PGA [m/s2]")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
    axes[0, 0].set_title("Sampling OOD by PGA scale: MSE")
    axes[0, 1].set_title("Sampling OOD by PGA scale: R2")
    savefig(fig, output_dir / "sampling_ood_by_scale_mse_r2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot O3 sampling OOD results.")
    parser.add_argument("--run_root", default="6-code/O3-generalization/results/sampling_ood/runs")
    parser.add_argument("--output_dir", default="6-code/O3-generalization/results/sampling_ood/figures")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    df = load_overall(run_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "sampling_ood_overall_summary.csv", index=False)
    plot_overall(df, output_dir)
    plot_cost(df, output_dir)
    plot_training_curves(run_root, output_dir)
    plot_by_scale(run_root, output_dir)
    print(f"Wrote sampling OOD figures to {output_dir}")


if __name__ == "__main__":
    main()
