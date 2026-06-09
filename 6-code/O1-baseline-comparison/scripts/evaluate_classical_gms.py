"""Evaluate O1 models on four classical ground-motion records.

Expected files under --test_sample_dir:
    1940-ElCentro-60s-50Hz-ms2.txt
    Blg1_1940-ElCentro-60s-50Hz-ms2_RoofAcc.txt
    1971-Sanfernando-60s-50Hz-ms2.txt
    Blg1_1971-Sanfernando-60s-50Hz-ms2_RoofAcc.txt
    1985-Michoacan-60s-200Hz-ms2.txt
    Blg1_1985-Michoacan-60s-200Hz-ms2_RoofAcc.txt
    1991-Kobe-60s-100Hz-ms2.txt
    Blg1_1991-Kobe-60s-100Hz-ms2_RoofAcc.txt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
O1_DIR = SCRIPT_DIR.parent
CODE_DIR = O1_DIR.parent
PROJECT_ROOT = CODE_DIR.parent
UPSTREAM_DIR = CODE_DIR / "shared" / "upstream" / "Seismic-FNO-clean"
UPSTREAM_MODULE_DIR = UPSTREAM_DIR / "module"
PROJECT_MODULE_DIR = PROJECT_ROOT / "module"
UTILS_DIR = CODE_DIR / "shared" / "utils"
for path in (SCRIPT_DIR, UPSTREAM_DIR, UPSTREAM_MODULE_DIR, PROJECT_ROOT, PROJECT_MODULE_DIR, UTILS_DIR):
    if path.exists():
        sys.path.insert(0, str(path))

from baseline_models import BASELINE_REGISTRY, build_model  # noqa: E402

GM_TO_GT_MAP = {
    "1940-ElCentro-60s-50Hz-ms2.txt": "Blg1_1940-ElCentro-60s-50Hz-ms2_RoofAcc.txt",
    "1971-Sanfernando-60s-50Hz-ms2.txt": "Blg1_1971-Sanfernando-60s-50Hz-ms2_RoofAcc.txt",
    "1985-Michoacan-60s-200Hz-ms2.txt": "Blg1_1985-Michoacan-60s-200Hz-ms2_RoofAcc.txt",
    "1991-Kobe-60s-100Hz-ms2.txt": "Blg1_1991-Kobe-60s-100Hz-ms2_RoofAcc.txt",
}

GM_TITLES = {
    "1940-ElCentro-60s-50Hz-ms2.txt": "El Centro (1940)",
    "1971-Sanfernando-60s-50Hz-ms2.txt": "San Fernando (1971)",
    "1985-Michoacan-60s-200Hz-ms2.txt": "Michoacan (1985)",
    "1991-Kobe-60s-100Hz-ms2.txt": "Kobe (1991)",
}

ZOOM_INTERVALS = {
    "1940-ElCentro-60s-50Hz-ms2.txt": (0.0, 10.0),
    "1971-Sanfernando-60s-50Hz-ms2.txt": (0.0, 6.0),
    "1985-Michoacan-60s-200Hz-ms2.txt": (8.0, 14.0),
    "1991-Kobe-60s-100Hz-ms2.txt": (2.0, 12.0),
}


def load_txt_data(file_path: Path, target_hz: int = 50, target_len: int = 3000) -> np.ndarray:
    times, values = [], []
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                times.append(float(parts[0]))
                values.append(float(parts[1]))
            except ValueError:
                continue
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(times) < 2:
        return np.zeros(target_len, dtype=np.float32)
    dt = times[1] - times[0]
    current_hz = round(1.0 / dt)
    step = max(1, int(current_hz / target_hz))
    out = values[::step]
    if len(out) > target_len:
        out = out[:target_len]
    elif len(out) < target_len:
        out = np.pad(out, (0, target_len - len(out)))
    return out.astype(np.float32)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.sum((y_true - y_pred) ** 2)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - err / denom) if denom > 0 else float("nan")


def load_baseline(model_name: str, checkpoint: Path, device: str):
    model = build_model(model_name)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def load_fno(checkpoint: Path, device: str):
    from neuralop.models import FNO

    model = FNO(
        n_modes=(512,),
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        projection_channel_ratio=2,
        n_layers=8,
        domain_padding=0.1,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model.to(device).eval()


def infer(model, gm: np.ndarray, device: str) -> tuple[np.ndarray, float]:
    x = torch.from_numpy(gm).float().unsqueeze(0).unsqueeze(0).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        pred = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return pred.detach().cpu().numpy().squeeze().astype(np.float64), time.time() - start


def spectral_r2(gt: np.ndarray, pred: np.ndarray, dt: float) -> float:
    gt_f = np.abs(np.fft.rfft(gt)) / len(gt)
    pred_f = np.abs(np.fft.rfft(pred)) / len(pred)
    return r2_score_np(gt_f, pred_f)


def plot_time_history(records: list[dict], output: Path, dt: float = 0.02) -> None:
    if not records:
        return
    models = sorted({r["model"] for r in records})
    events = list(GM_TO_GT_MAP)
    fig, axes = plt.subplots(len(events), 1, figsize=(12, 2.8 * len(events)), constrained_layout=True)
    if len(events) == 1:
        axes = [axes]
    for ax, event in zip(axes, events):
        rows = [r for r in records if r["filename"] == event]
        if not rows:
            ax.set_visible(False)
            continue
        t = np.arange(len(rows[0]["gt"])) * dt
        ax.plot(t, rows[0]["gt"], color="black", lw=1.2, label="Ground truth")
        for r in rows:
            ax.plot(t, r["pred"], lw=0.8, label=f"{r['model']} R2={r['r2_time']:.3f}")
        ax.set_title(GM_TITLES[event])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Roof acc. [m/s2]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=3)
    fig.savefig(output, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_metric_bars(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    import pandas as pd

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in {"gm", "gt", "pred"}} for r in rows])
    metrics = [("r2_time", "Time-history R2"), ("rmse_time", "Time RMSE"), ("r2_spectral", "Spectral R2"), ("pfa_abs_error_g", "PFA abs. error [g]")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (col, title) in zip(axes.ravel(), metrics):
        pivot = df.pivot(index="event", columns="model", values=col)
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    fig.savefig(output, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate O1 models on classical GMs.")
    parser.add_argument("--test_sample_dir", default="test_sample")
    parser.add_argument("--run_root", default=str(O1_DIR / "results" / "runs"))
    parser.add_argument("--output_dir", default=str(O1_DIR / "results" / "classical_gms"))
    parser.add_argument("--models", nargs="+", default=["MLP", "BiLSTM", "ResCNN1D", "UNet1D", "Transformer", "FNO-Large"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_hz", type=int, default=50)
    parser.add_argument("--target_len", type=int, default=3000)
    args = parser.parse_args()

    test_dir = Path(args.test_sample_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path(args.run_root)

    missing = []
    samples = []
    for gm_file, gt_file in GM_TO_GT_MAP.items():
        gm_path = test_dir / gm_file
        gt_path = test_dir / gt_file
        if not gm_path.exists() or not gt_path.exists():
            missing.append(str(gm_path if not gm_path.exists() else gt_path))
            continue
        samples.append({
            "filename": gm_file,
            "event": GM_TITLES[gm_file],
            "gm": load_txt_data(gm_path, args.target_hz, args.target_len),
            "gt": load_txt_data(gt_path, args.target_hz, args.target_len),
        })
    if missing:
        print("Missing classical GM files:")
        for item in missing:
            print("  ", item)
    if not samples:
        raise SystemExit("No classical GM records loaded.")

    rows = []
    for model_name in args.models:
        if model_name == "FNO-Large":
            ckpt = run_root / "FNO-Large-bs64" / "checkpoints" / "FNO-Large-bs64_best.pt"
            if not ckpt.exists():
                print(f"Skipping FNO-Large, checkpoint not found: {ckpt}")
                continue
            model = load_fno(ckpt, args.device)
        else:
            if model_name not in BASELINE_REGISTRY:
                print(f"Skipping unknown model: {model_name}")
                continue
            ckpt = run_root / model_name / "checkpoints" / f"{model_name}_best.pt"
            if not ckpt.exists():
                print(f"Skipping {model_name}, checkpoint not found: {ckpt}")
                continue
            model = load_baseline(model_name, ckpt, args.device)

        for sample in samples:
            pred, latency_s = infer(model, sample["gm"], args.device)
            gt = sample["gt"].astype(np.float64)
            rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
            mae = float(np.mean(np.abs(gt - pred)))
            pfa_gt = float(np.max(np.abs(gt)) / 9.81)
            pfa_pred = float(np.max(np.abs(pred)) / 9.81)
            row = {
                "model": model_name,
                "filename": sample["filename"],
                "event": sample["event"],
                "r2_time": r2_score_np(gt, pred),
                "rmse_time": rmse,
                "mae_time": mae,
                "r2_spectral": spectral_r2(gt, pred, 1.0 / args.target_hz),
                "pfa_gt_g": pfa_gt,
                "pfa_pred_g": pfa_pred,
                "pfa_abs_error_g": abs(pfa_gt - pfa_pred),
                "latency_ms": latency_s * 1000.0,
                "gm": sample["gm"],
                "gt": gt,
                "pred": pred,
            }
            rows.append(row)
            print(f"{model_name:12s} {sample['event']:24s} R2={row['r2_time']:.4f} RMSE={rmse:.4f}")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    serial_rows = [{k: v for k, v in row.items() if k not in {"gm", "gt", "pred"}} for row in rows]
    metrics_csv = output_dir / "classical_gm_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(serial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(serial_rows)
    (output_dir / "classical_gm_metrics.json").write_text(json.dumps(serial_rows, indent=2), encoding="utf-8")
    plot_time_history(rows, output_dir / "classical_time_histories.png", 1.0 / args.target_hz)
    plot_metric_bars(rows, output_dir / "classical_metric_bars.png")
    print(f"Wrote classical GM results to {output_dir}")


if __name__ == "__main__":
    main()
