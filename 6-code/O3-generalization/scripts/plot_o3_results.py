"""Plot O3 classical-GM and PGA-OOD review figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
O3_DIR = SCRIPT_DIR.parent
CODE_DIR = O3_DIR.parent
PROJECT_ROOT = CODE_DIR.parent
O1_SCRIPT_DIR = CODE_DIR / "O1-baseline-comparison" / "scripts"
UPSTREAM_DIR = CODE_DIR / "shared" / "upstream" / "Seismic-FNO-clean"
UPSTREAM_MODULE_DIR = UPSTREAM_DIR / "module"
PROJECT_MODULE_DIR = PROJECT_ROOT / "module"
UTILS_DIR = CODE_DIR / "shared" / "utils"
for path in (SCRIPT_DIR, O1_SCRIPT_DIR, UPSTREAM_DIR, UPSTREAM_MODULE_DIR, PROJECT_ROOT, PROJECT_MODULE_DIR, UTILS_DIR):
    if path.exists():
        sys.path.insert(0, str(path))

from dataprep_v2 import DynamicDataset  # type: ignore  # noqa: E402
from evaluate_baselines import move_batch  # noqa: E402
from splits import make_fixed_474_split  # noqa: E402
from train_eval_pga_ood import flatten, load_best_model  # noqa: E402


MODEL_ORDER = ["FNO-Large", "Transformer", "BiLSTM"]
MODEL_COLORS = {
    "Ground truth": "#111111",
    "FNO-Large": "#1f77b4",
    "Transformer": "#d62728",
    "BiLSTM": "#2ca02c",
}


def savefig(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(run_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    for model in MODEL_ORDER:
        log_path = run_root / model / "PGA-1to10" / "training_log.csv"
        df = pd.read_csv(log_path)
        axes[0].plot(df["epoch"], df["train_mse"], color=MODEL_COLORS[model], ls="--", lw=1.4, alpha=0.8, label=f"{model} train")
        axes[0].plot(df["epoch"], df["val_mse"], color=MODEL_COLORS[model], lw=2.0, label=f"{model} val")
        axes[1].plot(df["epoch"], df["val_r2"], color=MODEL_COLORS[model], lw=2.0, label=model)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title("PGA-OOD training and validation loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation R2")
    axes[1].set_title("Validation fidelity inside 1-10 m/s2 PGA range")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    savefig(fig, output_dir / "pga_ood_training_curves")


def plot_pga_degradation(run_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    for model in MODEL_ORDER:
        df = pd.read_csv(run_root / model / "PGA-1to10" / "evaluation_by_scale.csv")
        axes[0].plot(df["median_pga_ms2"], df["mse"], marker="o", ms=3.2, lw=2.0, color=MODEL_COLORS[model], label=model)
        axes[1].plot(df["median_pga_ms2"], df["r2"], marker="o", ms=3.2, lw=2.0, color=MODEL_COLORS[model], label=model)
    for ax in axes:
        ax.axvspan(1.0, 10.0, color="#d9ead3", alpha=0.55, label="Train PGA range")
        ax.set_xlabel("Median PGA of scale [m/s2]")
        ax.grid(True, alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_title("PGA extrapolation degradation: MSE")
    axes[1].set_ylabel("Test R2")
    axes[1].set_title("PGA extrapolation degradation: R2")
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[1].legend(unique.values(), unique.keys(), fontsize=8, loc="lower left")
    savefig(fig, output_dir / "pga_ood_by_scale_mse_r2")


def plot_cost_summary(run_root: Path, output_dir: Path) -> None:
    rows = []
    for model in MODEL_ORDER:
        df = pd.read_csv(run_root / model / "PGA-1to10" / "evaluation_overall.csv")
        rows.append(df.iloc[0].to_dict())
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), constrained_layout=True)
    metrics = [
        ("mse", "Overall test MSE"),
        ("train_time_s", "Training time [s]"),
        ("peak_gpu_gb", "Peak GPU [GB]"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(df["model"], df[col], color=[MODEL_COLORS[m] for m in df["model"]], width=0.72)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=18)
        if col == "mse":
            ax.set_yscale("log")
    savefig(fig, output_dir / "pga_ood_overall_cost_summary")


def load_models(run_root: Path, device: str) -> dict[str, torch.nn.Module]:
    models = {}
    for model in MODEL_ORDER:
        ckpt = run_root / model / "PGA-1to10" / "checkpoints" / f"{model}_PGA-1to10_best.pt"
        models[model] = load_best_model(model, ckpt, device)
    return models


def predict_examples(run_root: Path, output_dir: Path, base_data_dir: Path, building_dir: Path, device: str) -> None:
    split = make_fixed_474_split(train_gms_used=3000, use_scales_count=57, seed=42)
    selected_gms = [int(split.test_gms[0]), int(split.test_gms[1]), int(split.test_gms[2])]
    selected_scales = [0, 9, 46, 56]
    example_indices = flatten(np.asarray(selected_gms), np.asarray(selected_scales))
    gm_path = base_data_dir / "MDOF" / "All_GMs" / "GMs_knet_3474_AF_57.h5"
    ds = DynamicDataset(str(gm_path), str(building_dir), gm_indices=example_indices)

    models = load_models(run_root, device)
    records = []
    with torch.no_grad():
        for local_idx, global_idx in enumerate(example_indices):
            batch = ds[local_idx]
            x, y = move_batch([t.unsqueeze(0) if torch.is_tensor(t) else t for t in batch], device)
            gt = y.detach().cpu().numpy().squeeze()
            row = {
                "global_idx": int(global_idx),
                "gm_id": int(global_idx // 57),
                "scale_id": int(global_idx % 57),
                "gt": gt,
                "pred": {},
                "mse": {},
                "r2": {},
            }
            denom = float(np.sum((gt - np.mean(gt)) ** 2))
            for model_name, model in models.items():
                pred = model(x).detach().cpu().numpy().squeeze()
                row["pred"][model_name] = pred
                row["mse"][model_name] = float(np.mean((gt - pred) ** 2))
                row["r2"][model_name] = float(1.0 - np.sum((gt - pred) ** 2) / denom) if denom > 0 else float("nan")
            records.append(row)

    for model in models.values():
        del model
    if device == "cuda":
        torch.cuda.empty_cache()

    serial = [
        {
            "global_idx": r["global_idx"],
            "gm_id": r["gm_id"],
            "scale_id": r["scale_id"],
            **{f"{m}_mse": r["mse"][m] for m in MODEL_ORDER},
            **{f"{m}_r2": r["r2"][m] for m in MODEL_ORDER},
        }
        for r in records
    ]
    pd.DataFrame(serial).to_csv(output_dir / "pga_ood_example_metrics.csv", index=False)
    (output_dir / "pga_ood_example_metrics.json").write_text(json.dumps(serial, indent=2), encoding="utf-8")

    t = np.arange(records[0]["gt"].shape[-1]) * 0.02
    fig, axes = plt.subplots(len(records), 1, figsize=(13.0, 2.45 * len(records)), constrained_layout=True)
    if len(records) == 1:
        axes = [axes]
    for ax, rec in zip(axes, records):
        ax.plot(t, rec["gt"], color=MODEL_COLORS["Ground truth"], lw=1.2, label="Ground truth")
        for model in MODEL_ORDER:
            ax.plot(t, rec["pred"][model], color=MODEL_COLORS[model], lw=0.9, label=f"{model} R2={rec['r2'][model]:.3f}")
        train_tag = "train-range" if 9 <= rec["scale_id"] <= 46 else "OOD-scale"
        ax.set_title(f"GM {rec['gm_id']} / scale {rec['scale_id']} ({train_tag})")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Roof acc. [m/s2]")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=7, ncol=4)
    savefig(fig, output_dir / "pga_ood_test_examples_time_histories")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4), constrained_layout=True)
    subset = [r for r in records if r["scale_id"] in {0, 9, 46, 56}][:4]
    for ax, rec in zip(axes.ravel(), subset):
        start, end = 0, min(850, len(t))
        ax.plot(t[start:end], rec["gt"][start:end], color=MODEL_COLORS["Ground truth"], lw=1.4, label="GT")
        for model in MODEL_ORDER:
            ax.plot(t[start:end], rec["pred"][model][start:end], color=MODEL_COLORS[model], lw=0.95, label=model)
        ax.set_title(f"GM {rec['gm_id']} / scale {rec['scale_id']}")
        ax.grid(True, alpha=0.22)
    axes.ravel()[0].legend(fontsize=7, ncol=4)
    savefig(fig, output_dir / "pga_ood_test_examples_zoom")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create O3 review figures.")
    parser.add_argument("--pga_run_root", default=str(O3_DIR / "results" / "pga_ood" / "runs"))
    parser.add_argument("--output_dir", default=str(O3_DIR / "results" / "figures_o3_review"))
    parser.add_argument("--base_data_dir", default="/data/home/jason/data/SesimicTransformerData")
    parser.add_argument("--building_dir", default="/data/home/jason/data/SesimicTransformerData/MDOF/knet-250/Data/fno")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_examples", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path(args.pga_run_root)
    plot_training_curves(run_root, output_dir)
    plot_pga_degradation(run_root, output_dir)
    plot_cost_summary(run_root, output_dir)
    if not args.skip_examples:
        predict_examples(run_root, output_dir, Path(args.base_data_dir), Path(args.building_dir), args.device)
    print(f"Wrote O3 review figures to {output_dir}")


if __name__ == "__main__":
    main()
