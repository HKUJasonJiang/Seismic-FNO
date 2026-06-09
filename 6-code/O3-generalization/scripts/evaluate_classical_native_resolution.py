"""Evaluate O1 checkpoints on classical GMs using native GM input resolution.

The current classical response files in data/4_ClassicalGMs are 50Hz roof
responses even for Kobe/Michoacan. Therefore this script keeps each GM input at
its native resolution, predicts at that native input length, then interpolates
the prediction to the available roof-response time axis for metric computation.
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
import torch.nn.functional as F


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

from baseline_models import PatchTransformer, build_model  # noqa: E402


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

COLORS = {
    "Ground truth": "#111111",
    "BiLSTM": "#2ca02c",
    "Transformer": "#d62728",
    "FNO-Large": "#1f77b4",
}


def load_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                times.append(float(parts[0]))
                values.append(float(parts[1]))
            except ValueError:
                continue
    if not times:
        raise ValueError(f"No numeric data loaded from {path}")
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float32)


def infer(model: torch.nn.Module, gm: np.ndarray, device: str) -> tuple[np.ndarray, float]:
    x = torch.from_numpy(gm).float().unsqueeze(0).unsqueeze(0).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        pred = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return pred.detach().cpu().numpy().squeeze().astype(np.float64), time.time() - start


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.sum((y_true - y_pred) ** 2)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - err / denom) if denom > 0 else float("nan")


def spectral_log_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_fft = np.abs(np.fft.rfft(y_true))
    pred_fft = np.abs(np.fft.rfft(y_pred))
    return float(np.mean((np.log1p(true_fft) - np.log1p(pred_fft)) ** 2))


def load_bilstm(checkpoint: Path, device: str) -> torch.nn.Module:
    model = build_model("BiLSTM")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def load_fno(checkpoint: Path, device: str) -> torch.nn.Module:
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


def interpolate_pos_embed(pos_embed: torch.Tensor, target_patches: int) -> torch.Tensor:
    if pos_embed.shape[1] == target_patches:
        return pos_embed
    src = pos_embed.permute(0, 2, 1)
    dst = F.interpolate(src, size=target_patches, mode="linear", align_corners=True)
    return dst.permute(0, 2, 1)


def load_transformer(checkpoint: Path, device: str, seq_len: int) -> torch.nn.Module:
    model = PatchTransformer(seq_len=seq_len)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    if "pos_embed" in state:
        target_patches = seq_len // model.patch_size
        state = dict(state)
        state["pos_embed"] = interpolate_pos_embed(state["pos_embed"], target_patches)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def model_checkpoint(run_root: Path, model: str) -> Path:
    if model == "FNO-Large":
        return run_root / "FNO-Large-bs64" / "checkpoints" / "FNO-Large-bs64_best.pt"
    return run_root / model / "checkpoints" / f"{model}_best.pt"


def load_model(model: str, checkpoint: Path, device: str, seq_len: int) -> torch.nn.Module:
    if model == "BiLSTM":
        return load_bilstm(checkpoint, device)
    if model == "Transformer":
        return load_transformer(checkpoint, device, seq_len)
    if model == "FNO-Large":
        return load_fno(checkpoint, device)
    raise ValueError(model)


def plot_time_histories(records: list[dict], output: Path) -> None:
    events = list(GM_TO_GT_MAP)
    fig, axes = plt.subplots(len(events), 1, figsize=(13, 2.9 * len(events)), constrained_layout=True)
    if len(events) == 1:
        axes = [axes]
    for ax, gm_file in zip(axes, events):
        rows = [r for r in records if r["filename"] == gm_file]
        if not rows:
            ax.set_visible(False)
            continue
        ax.plot(rows[0]["gt_time"], rows[0]["gt"], color=COLORS["Ground truth"], lw=1.25, label="Ground truth (available 50Hz response)")
        for r in rows:
            ax.plot(
                r["gt_time"],
                r["pred_on_gt_time"],
                color=COLORS[r["model"]],
                lw=0.9,
                label=f"{r['model']} R2={r['r2_time']:.3f}",
            )
        native_hz = round(1.0 / r["gm_dt"])
        gt_hz = round(1.0 / r["gt_dt"])
        ax.set_title(f"{GM_TITLES[gm_file]}: native GM {native_hz}Hz, response GT {gt_hz}Hz")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Roof acc. [m/s2]")
        ax.grid(True, alpha=0.24)
        ax.legend(fontsize=7, ncol=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(rows: list[dict], output: Path) -> None:
    df = (
        __import__("pandas")
        .DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in rows])
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, (col, title) in zip(
        axes,
        [
            ("r2_time", "R2 on available response grid"),
            ("rmse_time", "RMSE"),
            ("pfa_abs_error_g", "PFA abs. error [g]"),
        ],
    ):
        pivot = df.pivot(index="event", columns="model", values=col)
        pivot.plot(kind="bar", ax=ax, color=[COLORS.get(c, "#999999") for c in pivot.columns], width=0.78)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.24)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classical GMs at native GM input resolution.")
    parser.add_argument("--classical_dir", default=str(PROJECT_ROOT / "data" / "4_ClassicalGMs"))
    parser.add_argument("--run_root", default=str(CODE_DIR / "O1-baseline-comparison" / "results" / "runs"))
    parser.add_argument("--output_dir", default=str(O3_DIR / "results" / "classical_gms_native_input_o1_full"))
    parser.add_argument("--models", nargs="+", default=["BiLSTM", "Transformer", "FNO-Large"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    classical_dir = Path(args.classical_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path(args.run_root)

    samples = []
    for gm_file, gt_file in GM_TO_GT_MAP.items():
        gm_time, gm = load_two_column(classical_dir / gm_file)
        gt_time, gt = load_two_column(classical_dir / gt_file)
        samples.append(
            {
                "filename": gm_file,
                "event": GM_TITLES[gm_file],
                "gm_time": gm_time,
                "gm": gm,
                "gt_time": gt_time,
                "gt": gt.astype(np.float64),
                "gm_dt": float(np.median(np.diff(gm_time))),
                "gt_dt": float(np.median(np.diff(gt_time))),
            }
        )

    rows = []
    for sample in samples:
        seq_len = len(sample["gm"])
        for model_name in args.models:
            ckpt = model_checkpoint(run_root, model_name)
            model = load_model(model_name, ckpt, args.device, seq_len)
            pred_native, latency_s = infer(model, sample["gm"], args.device)
            pred_on_gt_time = np.interp(sample["gt_time"], sample["gm_time"], pred_native)
            gt = sample["gt"]
            rmse = float(np.sqrt(np.mean((gt - pred_on_gt_time) ** 2)))
            mae = float(np.mean(np.abs(gt - pred_on_gt_time)))
            pfa_gt = float(np.max(np.abs(gt)) / 9.81)
            pfa_pred = float(np.max(np.abs(pred_on_gt_time)) / 9.81)
            row = {
                "model": model_name,
                "filename": sample["filename"],
                "event": sample["event"],
                "gm_native_hz": round(1.0 / sample["gm_dt"]),
                "gm_native_len": seq_len,
                "response_gt_hz": round(1.0 / sample["gt_dt"]),
                "response_gt_len": len(gt),
                "comparison": "native_input_prediction_interpolated_to_available_50hz_response_gt",
                "r2_time": r2_score(gt, pred_on_gt_time),
                "rmse_time": rmse,
                "mae_time": mae,
                "spectral_log_mse": spectral_log_mse(gt, pred_on_gt_time),
                "pfa_gt_g": pfa_gt,
                "pfa_pred_g": pfa_pred,
                "pfa_abs_error_g": abs(pfa_gt - pfa_pred),
                "latency_ms": latency_s * 1000.0,
                "gt_time": sample["gt_time"],
                "gt": gt,
                "pred_native_time": sample["gm_time"],
                "pred_native": pred_native,
                "pred_on_gt_time": pred_on_gt_time,
                "gm_dt": sample["gm_dt"],
                "gt_dt": sample["gt_dt"],
            }
            rows.append(row)
            print(
                f"{model_name:12s} {sample['event']:24s} "
                f"input={row['gm_native_hz']}Hz gt={row['response_gt_hz']}Hz "
                f"R2={row['r2_time']:.4f} RMSE={rmse:.4f}"
            )
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

    serial_rows = [
        {k: v for k, v in row.items() if k not in {"gt_time", "gt", "pred_native_time", "pred_native", "pred_on_gt_time"}}
        for row in rows
    ]
    with (output_dir / "classical_native_input_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(serial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(serial_rows)
    (output_dir / "classical_native_input_metrics.json").write_text(json.dumps(serial_rows, indent=2), encoding="utf-8")
    plot_time_histories(rows, output_dir / "classical_native_input_time_histories.png")
    plot_metric_bars(rows, output_dir / "classical_native_input_metric_bars.png")
    note = {
        "important_caveat": (
            "GM inputs are kept at native 50/100/200Hz resolution, but the current "
            "roof response files are all 50Hz. Metrics are computed after interpolating "
            "native-length predictions to the available 50Hz response time axis."
        ),
        "output_dir": str(output_dir),
    }
    (output_dir / "README.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(f"Wrote native-input classical results to {output_dir}")


if __name__ == "__main__":
    main()
