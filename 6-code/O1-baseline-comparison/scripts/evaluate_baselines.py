"""Evaluate O1 baselines and optional FNO-Large on the fixed test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
O1_DIR = SCRIPT_DIR.parent
CODE_DIR = O1_DIR.parent
UPSTREAM_DIR = CODE_DIR / "shared" / "upstream" / "Seismic-FNO-clean"
UPSTREAM_MODULE_DIR = UPSTREAM_DIR / "module"
UTILS_DIR = CODE_DIR / "shared" / "utils"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(UPSTREAM_DIR))
sys.path.insert(0, str(UPSTREAM_MODULE_DIR))
sys.path.insert(0, str(UTILS_DIR))

from baseline_models import BASELINE_REGISTRY, build_model, count_parameters  # noqa: E402
from dataprep_v2 import DynamicDataset  # type: ignore  # noqa: E402
from metrics import PFAMeter, RegressionMeter  # noqa: E402
from splits import make_fixed_474_split  # noqa: E402


DEFAULT_BASE_DATA = Path(r"D:\BaiduNetdiskDownload\SesimicTransformerData")


def move_batch(batch, device: str):
    gm_data, _attrs, acc_response, _damage = batch
    x = gm_data.permute(0, 2, 1).to(device, non_blocking=True)
    y = acc_response.permute(0, 2, 1).to(device, non_blocking=True)
    return x, y


def compute_spectral_metrics(pred: torch.Tensor, target: torch.Tensor, dt: float = 0.02) -> dict[str, float]:
    pred_fft = torch.abs(torch.fft.rfft(pred.detach(), dim=-1))
    target_fft = torch.abs(torch.fft.rfft(target.detach(), dim=-1))
    log_err = torch.mean((torch.log1p(pred_fft) - torch.log1p(target_fft)) ** 2)
    freqs = torch.fft.rfftfreq(pred.shape[-1], d=dt).to(pred.device)
    high = freqs >= 5.0
    high_err = torch.mean((torch.log1p(pred_fft[..., high]) - torch.log1p(target_fft[..., high])) ** 2)
    return {"spectral_log_mse": float(log_err.item()), "high_freq_log_mse": float(high_err.item())}


def evaluate_model(model, loader, device: str, latency_batches: int = 20) -> dict:
    model.eval()
    criterion = torch.nn.MSELoss()
    reg = RegressionMeter()
    pfa = PFAMeter()
    spectral_sum = 0.0
    high_sum = 0.0
    n_batches = 0
    latency_times = []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            x, y = move_batch(batch, device)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            pred = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            if n_batches < latency_batches:
                latency_times.append(elapsed / x.shape[0])
            loss = criterion(pred, y)
            loss_sum += float(loss.item())
            reg.update(pred, y)
            pfa.update(pred, y)
            spec = compute_spectral_metrics(pred, y)
            spectral_sum += spec["spectral_log_mse"]
            high_sum += spec["high_freq_log_mse"]
            n_batches += 1
    metrics = reg.compute()
    metrics.update(pfa.compute())
    metrics["loss"] = loss_sum / max(n_batches, 1)
    metrics["spectral_log_mse"] = spectral_sum / max(n_batches, 1)
    metrics["high_freq_log_mse"] = high_sum / max(n_batches, 1)
    metrics["latency_ms_per_sample"] = float(np.mean(latency_times) * 1000.0) if latency_times else float("nan")
    return metrics


def load_baseline(model_name: str, checkpoint_path: Path, device: str):
    model = build_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def load_fno_large(checkpoint_path: Path, device: str):
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model.to(device)


def find_checkpoint(run_root: Path, model_name: str) -> Path | None:
    candidates = [run_root / model_name / "checkpoints" / f"{model_name}_best.pt"]
    model_dir = run_root / model_name
    if model_dir.exists():
        candidates.extend(sorted(model_dir.glob("**/*best*.pt")))
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate O1 baselines.")
    parser.add_argument("--models", nargs="+", default=list(BASELINE_REGISTRY))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_data_dir", default=str(DEFAULT_BASE_DATA))
    parser.add_argument("--building_dir", default=None)
    parser.add_argument("--run_root", default=str(O1_DIR / "results" / "runs"))
    parser.add_argument("--fno_checkpoint", default=None)
    parser.add_argument("--output", default=str(O1_DIR / "results" / "evaluation" / "table4_metrics.csv"))
    args = parser.parse_args()

    base_data = Path(args.base_data_dir)
    gm_path = base_data / "MDOF" / "All_GMs" / "GMs_knet_3474_AF_57.h5"
    building_dir = Path(args.building_dir) if args.building_dir else base_data / "MDOF" / "knet-250" / "Data" / "fno"
    split = make_fixed_474_split(use_scales_count=57, seed=42)
    test_ds = DynamicDataset(str(gm_path), str(building_dir), gm_indices=split.test_indices)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    rows = []
    run_root = Path(args.run_root)
    for model_name in args.models:
        ckpt = find_checkpoint(run_root, model_name)
        if ckpt is None:
            print(f"Skipping {model_name}: checkpoint not found under {run_root}")
            continue
        model = load_baseline(model_name, ckpt, args.device)
        params = count_parameters(model)
        metrics = evaluate_model(model, loader, args.device)
        row = {"model": model_name, "checkpoint": str(ckpt), **params, **metrics}
        rows.append(row)
        print(f"{model_name}: mse={metrics['mse']:.6f}, r2={metrics['r2']:.4f}, pfa_rmse={metrics['pfa_rmse_g']:.4f}g")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if args.fno_checkpoint:
        fno_path = Path(args.fno_checkpoint)
        model = load_fno_large(fno_path, args.device)
        params = count_parameters(model)
        metrics = evaluate_model(model, loader, args.device)
        row = {"model": "FNO-Large", "checkpoint": str(fno_path), **params, **metrics}
        rows.append(row)
        print(f"FNO-Large: mse={metrics['mse']:.6f}, r2={metrics['r2']:.4f}, pfa_rmse={metrics['pfa_rmse_g']:.4f}g")

    if not rows:
        print("No models evaluated.")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output.with_suffix(".json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Evaluation table saved to {output}")


if __name__ == "__main__":
    main()
