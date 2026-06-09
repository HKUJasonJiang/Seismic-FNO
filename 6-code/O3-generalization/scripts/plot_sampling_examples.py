"""Plot fixed-test example predictions for O3 sampling-resolution OOD."""

from __future__ import annotations

import argparse
import csv
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
from train_eval_sampling_ood import load_best_model  # noqa: E402


MODELS = ["FNO-Large", "Transformer", "BiLSTM"]
RUNS = ["S-20Hz", "S-25Hz", "S-50Hz"]
RUN_COLORS = {"S-20Hz": "#ff7f0e", "S-25Hz": "#9467bd", "S-50Hz": "#1f77b4"}


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.sum((y_true - y_pred) ** 2)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - err / denom) if denom > 0 else float("nan")


def load_models(run_root: Path, device: str) -> dict[tuple[str, str], torch.nn.Module]:
    models = {}
    for model_name in MODELS:
        for run in RUNS:
            ckpt = run_root / model_name / run / "checkpoints" / f"{model_name}_{run}_best.pt"
            models[(model_name, run)] = load_best_model(model_name, ckpt, device, eval_seq_len=3000)
    return models


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sampling OOD fixed-test examples.")
    parser.add_argument("--run_root", default=str(O3_DIR / "results" / "sampling_ood" / "runs"))
    parser.add_argument("--output_dir", default=str(O3_DIR / "results" / "sampling_ood" / "figures"))
    parser.add_argument("--base_data_dir", default="/data/home/jason/data/SesimicTransformerData")
    parser.add_argument("--building_dir", default="/data/home/jason/data/SesimicTransformerData/MDOF/knet-250/Data/fno")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    base_data = Path(args.base_data_dir)
    gm_path = base_data / "MDOF" / "All_GMs" / "GMs_knet_3474_AF_57.h5"
    building_dir = Path(args.building_dir)
    split = make_fixed_474_split(train_gms_used=3000, use_scales_count=57, seed=42)
    sample_specs = [
        (int(split.test_gms[0]), 9),
        (int(split.test_gms[1]), 46),
        (int(split.test_gms[2]), 56),
    ]
    scale_map = pd.read_csv(run_root / "FNO-Large" / "S-20Hz" / "scale_pga_mapping.csv").set_index("scale_id")
    models = load_models(run_root, args.device)

    records = []
    for gm_id, scale_id in sample_specs:
        global_idx = gm_id * 57 + scale_id
        ds = DynamicDataset(str(gm_path), str(building_dir), gm_indices=np.asarray([global_idx]))
        batch = ds[0]
        x, y = move_batch([item.unsqueeze(0) if torch.is_tensor(item) else item for item in batch], args.device)
        gt = y.detach().cpu().numpy().squeeze()
        pred_map = {}
        metric_map = {}
        with torch.no_grad():
            for model_name in MODELS:
                for run in RUNS:
                    pred = models[(model_name, run)](x).detach().cpu().numpy().squeeze()
                    pred_map[(model_name, run)] = pred
                    metric_map[(model_name, run)] = {
                        "mse": float(np.mean((gt - pred) ** 2)),
                        "r2": r2_score(gt, pred),
                    }
        records.append(
            {
                "gm_id": gm_id,
                "scale_id": scale_id,
                "global_idx": global_idx,
                "median_pga_ms2": float(scale_map.loc[scale_id, "median_pga_ms2"]),
                "gt": gt,
                "pred": pred_map,
                "metrics": metric_map,
            }
        )

    for model in models.values():
        del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    rows = []
    for rec in records:
        for model_name in MODELS:
            for run in RUNS:
                rows.append(
                    {
                        "gm_id": rec["gm_id"],
                        "scale_id": rec["scale_id"],
                        "global_idx": rec["global_idx"],
                        "median_pga_ms2": rec["median_pga_ms2"],
                        "model": model_name,
                        "run_id": run,
                        **rec["metrics"][(model_name, run)],
                    }
                )
    with (output_dir / "sampling_ood_example_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    t = np.arange(3000) * 0.02
    fig, axes = plt.subplots(len(records), len(MODELS), figsize=(15.6, 8.6), constrained_layout=True)
    for row_idx, rec in enumerate(records):
        for col_idx, model_name in enumerate(MODELS):
            ax = axes[row_idx, col_idx]
            ax.plot(t, rec["gt"], color="black", lw=1.15, label="GT 50Hz")
            for run in RUNS:
                m = rec["metrics"][(model_name, run)]
                ax.plot(
                    t,
                    rec["pred"][(model_name, run)],
                    color=RUN_COLORS[run],
                    lw=0.85,
                    label=f"{run} R2={m['r2']:.2f}",
                )
            ax.set_title(f"{model_name}: GM {rec['gm_id']} scale {rec['scale_id']} ({rec['median_pga_ms2']:.1f} m/s2)")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Roof acc. [m/s2]")
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=7, ncol=2)
    savefig(fig, output_dir / "sampling_ood_test_examples_time_histories")

    fig, axes = plt.subplots(len(records), len(MODELS), figsize=(15.6, 8.6), constrained_layout=True)
    for row_idx, rec in enumerate(records):
        gt = rec["gt"]
        peak = int(np.argmax(np.abs(gt)))
        start = max(0, peak - 220)
        end = min(len(gt), peak + 520)
        for col_idx, model_name in enumerate(MODELS):
            ax = axes[row_idx, col_idx]
            ax.plot(t[start:end], gt[start:end], color="black", lw=1.25, label="GT 50Hz")
            for run in RUNS:
                m = rec["metrics"][(model_name, run)]
                ax.plot(
                    t[start:end],
                    rec["pred"][(model_name, run)][start:end],
                    color=RUN_COLORS[run],
                    lw=0.95,
                    label=f"{run} R2={m['r2']:.2f}",
                )
            ax.set_title(f"{model_name}: peak zoom, GM {rec['gm_id']} scale {rec['scale_id']}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Roof acc. [m/s2]")
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=7, ncol=2)
    savefig(fig, output_dir / "sampling_ood_test_examples_peak_zoom")
    print(f"Wrote sampling OOD example figures to {output_dir}")


if __name__ == "__main__":
    main()
