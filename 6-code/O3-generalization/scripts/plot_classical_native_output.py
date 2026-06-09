"""Plot four classical GM native-input/native-output predictions.

Predictions are generated at each GM file's native sampling rate. The currently
available roof-response files are 50Hz; for 100/200Hz events, the GT curve is
interpolated to the native GM time axis for visualization and metric reporting.
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

COLORS = {"GT": "#111111", "BiLSTM": "#2ca02c", "Transformer": "#d62728", "FNO-Large": "#1f77b4"}
MODELS = ["BiLSTM", "Transformer", "FNO-Large"]


def load_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times, values = [], []
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


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.sum((y_true - y_pred) ** 2)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - err / denom) if denom > 0 else float("nan")


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
    state = dict(ckpt["model_state_dict"])
    target_patches = seq_len // model.patch_size
    state["pos_embed"] = interpolate_pos_embed(state["pos_embed"], target_patches)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def checkpoint_for(run_root: Path, model_name: str) -> Path:
    if model_name == "FNO-Large":
        return run_root / "FNO-Large-bs64" / "checkpoints" / "FNO-Large-bs64_best.pt"
    return run_root / model_name / "checkpoints" / f"{model_name}_best.pt"


def load_model(run_root: Path, model_name: str, seq_len: int, device: str) -> torch.nn.Module:
    ckpt = checkpoint_for(run_root, model_name)
    if model_name == "BiLSTM":
        return load_bilstm(ckpt, device)
    if model_name == "Transformer":
        return load_transformer(ckpt, device, seq_len)
    if model_name == "FNO-Large":
        return load_fno(ckpt, device)
    raise ValueError(model_name)


def plot_event(event: dict, output_dir: Path) -> None:
    gm_time = event["gm_time"]
    gt_native = event["gt_on_native_time"]
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 6.6), constrained_layout=True, height_ratios=[2, 1])
    ax = axes[0]
    ax.plot(gm_time, gt_native, color=COLORS["GT"], lw=1.25, label="GT roof response (50Hz file interpolated to native time axis)")
    for model_name in MODELS:
        pred = event["pred"][model_name]
        m = event["metrics"][model_name]
        ax.plot(gm_time, pred, color=COLORS[model_name], lw=0.9, label=f"{model_name} R2={m['r2_native_axis']:.3f}")
    ax.set_title(f"{event['title']}: native GM/output {event['gm_hz']}Hz, available roof GT {event['gt_hz']}Hz")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Roof acc. [m/s2]")
    ax.grid(True, alpha=0.23)
    ax.legend(fontsize=8, ncol=4)

    peak = int(np.argmax(np.abs(gt_native)))
    start = max(0, peak - int(4.0 * event["gm_hz"]))
    end = min(len(gm_time), peak + int(8.0 * event["gm_hz"]))
    ax = axes[1]
    ax.plot(gm_time[start:end], gt_native[start:end], color=COLORS["GT"], lw=1.3, label="GT")
    for model_name in MODELS:
        pred = event["pred"][model_name]
        ax.plot(gm_time[start:end], pred[start:end], color=COLORS[model_name], lw=0.95, label=model_name)
    ax.set_title("Peak-window zoom")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Roof acc. [m/s2]")
    ax.grid(True, alpha=0.23)
    ax.legend(fontsize=8, ncol=4)

    stem = Path(event["filename"]).stem.replace("-60s", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}_native_output.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}_native_output.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot classical native-output model predictions.")
    parser.add_argument("--classical_dir", default=str(PROJECT_ROOT / "data" / "4_ClassicalGMs"))
    parser.add_argument("--run_root", default=str(CODE_DIR / "O1-baseline-comparison" / "results" / "runs"))
    parser.add_argument("--output_dir", default=str(O3_DIR / "results" / "classical_gms_native_output_o1_full"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    classical_dir = Path(args.classical_dir)
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for gm_file, gt_file in GM_TO_GT_MAP.items():
        gm_time, gm = load_two_column(classical_dir / gm_file)
        gt_time, gt = load_two_column(classical_dir / gt_file)
        gm_dt = float(np.median(np.diff(gm_time)))
        gt_dt = float(np.median(np.diff(gt_time)))
        gm_hz = int(round(1.0 / gm_dt))
        gt_hz = int(round(1.0 / gt_dt))
        gt_on_native = np.interp(gm_time, gt_time, gt.astype(np.float64))
        event = {
            "filename": gm_file,
            "title": GM_TITLES[gm_file],
            "gm_time": gm_time,
            "gm_hz": gm_hz,
            "gt_hz": gt_hz,
            "gt_on_native_time": gt_on_native,
            "pred": {},
            "metrics": {},
        }
        for model_name in MODELS:
            model = load_model(run_root, model_name, len(gm), args.device)
            pred_native, latency_s = infer(model, gm, args.device)
            mse = float(np.mean((gt_on_native - pred_native) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(gt_on_native - pred_native)))
            r2 = r2_score(gt_on_native, pred_native)
            pfa_gt = float(np.max(np.abs(gt_on_native)) / 9.81)
            pfa_pred = float(np.max(np.abs(pred_native)) / 9.81)
            event["pred"][model_name] = pred_native
            event["metrics"][model_name] = {"mse_native_axis": mse, "rmse_native_axis": rmse, "mae_native_axis": mae, "r2_native_axis": r2}
            rows.append({
                "model": model_name,
                "filename": gm_file,
                "event": GM_TITLES[gm_file],
                "gm_native_hz": gm_hz,
                "gm_native_len": len(gm),
                "available_gt_hz": gt_hz,
                "available_gt_len": len(gt),
                "comparison": "model_native_output_vs_available_50hz_gt_interpolated_to_native_time_axis",
                "mse_native_axis": mse,
                "rmse_native_axis": rmse,
                "mae_native_axis": mae,
                "r2_native_axis": r2,
                "pfa_gt_g_native_axis": pfa_gt,
                "pfa_pred_g_native_axis": pfa_pred,
                "pfa_abs_error_g_native_axis": abs(pfa_gt - pfa_pred),
                "latency_ms": latency_s * 1000.0,
            })
            print(f"{model_name:12s} {GM_TITLES[gm_file]:24s} output={gm_hz}Hz R2={r2:.4f} RMSE={rmse:.4f}")
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        plot_event(event, output_dir)

    with (output_dir / "classical_native_output_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "classical_native_output_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    note = {
        "caveat": "Model predictions are native-length outputs. Available roof-response GT files are 50Hz, so GT is interpolated to the native GM/output time axis for visualization and metrics.",
        "output_dir": str(output_dir),
    }
    (output_dir / "README.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(f"Wrote four native-output classical figures to {output_dir}")


if __name__ == "__main__":
    main()
