"""Train/evaluate O3 PGA extrapolation OOD runs.

Train on canonical 3,000-GM pool using only AC/PGA levels inside a target range
and evaluate on the unchanged fixed 474-GM x 57-scale test grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


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

from baseline_models import build_model, count_parameters  # noqa: E402
from dataprep_v2 import DynamicDataset  # type: ignore  # noqa: E402
from evaluate_baselines import evaluate_model  # noqa: E402
from splits import make_fixed_474_split, save_split  # noqa: E402
from train_baselines import NonFiniteTrainingError, run_epoch, save_checkpoint, set_seed  # noqa: E402


DEFAULT_BASE_DATA = Path("/data/home/jason/data/SesimicTransformerData")


def build_fno_large() -> torch.nn.Module:
    from neuralop.models import FNO

    return FNO(
        n_modes=(512,),
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        projection_channel_ratio=2,
        n_layers=8,
        domain_padding=0.1,
    )


def build_o3_model(model_name: str) -> torch.nn.Module:
    if model_name == "FNO-Large":
        return build_fno_large()
    return build_model(model_name)


def model_lr(model_name: str, args: argparse.Namespace) -> float:
    if model_name == "BiLSTM":
        return args.bilstm_lr
    if model_name == "Transformer":
        return args.transformer_lr
    return args.lr


def flatten(gms: np.ndarray, scales: np.ndarray, total_scales: int = 57) -> np.ndarray:
    gm_grid, scale_grid = np.meshgrid(gms, scales, indexing="ij")
    return (gm_grid * total_scales + scale_grid).reshape(-1)


def infer_scale_mapping(gm_path: Path) -> list[dict[str, float | int | bool]]:
    with h5py.File(gm_path, "r") as f:
        acc = f["Acc_GMs"]
        total_gms = acc.shape[0] // 57
        gm_ids = np.arange(total_gms)
        rows = []
        for scale_id in range(57):
            idx = gm_ids * 57 + scale_id
            pgas = np.max(np.abs(acc[idx]), axis=1)
            rows.append(
                {
                    "scale_id": scale_id,
                    "mean_pga_ms2": float(np.mean(pgas)),
                    "median_pga_ms2": float(np.median(pgas)),
                    "min_pga_ms2": float(np.min(pgas)),
                    "max_pga_ms2": float(np.max(pgas)),
                }
            )
    return rows


def select_train_scales(mapping: list[dict[str, float | int | bool]], min_pga: float, max_pga: float) -> np.ndarray:
    selected = [
        int(row["scale_id"])
        for row in mapping
        if float(row["median_pga_ms2"]) >= min_pga and float(row["median_pga_ms2"]) <= max_pga
    ]
    if not selected:
        raise ValueError(f"No scale IDs selected for PGA range [{min_pga}, {max_pga}].")
    return np.asarray(selected, dtype=int)


def load_best_model(model_name: str, checkpoint_path: Path, device: str) -> torch.nn.Module:
    model = build_o3_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=(model_name != "FNO-Large"))
    return model.to(device).eval()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate one O3 PGA extrapolation OOD model.")
    parser.add_argument("--model", required=True, choices=["FNO-Large", "Transformer", "BiLSTM"])
    parser.add_argument("--run_id", default="PGA-1to10")
    parser.add_argument("--min_train_pga", type=float, default=1.0)
    parser.add_argument("--max_train_pga", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bilstm_lr", type=float, default=3e-4)
    parser.add_argument("--transformer_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--scheduler_step", type=int, default=20)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_data_dir", default=str(DEFAULT_BASE_DATA))
    parser.add_argument("--building_dir", default=None)
    parser.add_argument("--output_root", default=str(O3_DIR / "results" / "pga_ood"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    output_root = Path(args.output_root)
    run_dir = output_root / "runs" / args.model / args.run_id
    eval_path = run_dir / "evaluation_overall.csv"
    if eval_path.exists() and not args.force:
        print(f"Skipping {args.model}/{args.run_id}: {eval_path} exists.")
        return

    base_data = Path(args.base_data_dir)
    gm_path = base_data / "MDOF" / "All_GMs" / "GMs_knet_3474_AF_57.h5"
    building_dir = Path(args.building_dir) if args.building_dir else base_data / "MDOF" / "knet-250" / "Data" / "fno"
    run_dir.mkdir(parents=True, exist_ok=True)

    mapping = infer_scale_mapping(gm_path)
    train_scales = select_train_scales(mapping, args.min_train_pga, args.max_train_pga)
    for row in mapping:
        row["in_train_pga_range"] = int(int(row["scale_id"]) in set(train_scales.tolist()))
    write_csv(run_dir / "scale_pga_mapping.csv", mapping)

    split = make_fixed_474_split(train_gms_used=3000, use_scales_count=57, seed=args.seed)
    train_indices = flatten(split.train_gms, train_scales)
    val_indices = flatten(split.val_gms, train_scales)
    test_indices = flatten(split.test_gms, np.arange(57))
    save_split(split, run_dir / "canonical_full_split_reference")

    train_ds = DynamicDataset(str(gm_path), str(building_dir), gm_indices=train_indices)
    val_ds = DynamicDataset(str(gm_path), str(building_dir), gm_indices=val_indices)
    test_ds = DynamicDataset(str(gm_path), str(building_dir), gm_indices=test_indices)
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_o3_model(args.model).to(device)
    stats = count_parameters(model)
    lr = model_lr(args.model, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = torch.nn.MSELoss()
    grad_clip = args.grad_clip if args.model in {"BiLSTM", "Transformer"} else 0.0

    config = {
        "model": args.model,
        "run_id": args.run_id,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": lr,
        "weight_decay": args.weight_decay,
        "scheduler_step": args.scheduler_step,
        "scheduler_gamma": args.scheduler_gamma,
        "grad_clip": grad_clip,
        "min_train_pga": args.min_train_pga,
        "max_train_pga": args.max_train_pga,
        "train_scale_indices": train_scales.tolist(),
        "train_gms": len(split.train_gms),
        "val_gms": len(split.val_gms),
        "test_gms": len(split.test_gms),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_indices),
        "params": stats,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    fieldnames = [
        "epoch", "train_mse", "train_rmse", "train_mae", "train_r2",
        "val_mse", "val_rmse", "val_mae", "val_r2", "val_pfa_rmse_g",
        "lr", "train_max_grad_norm", "epoch_s", "gpu_peak_gb",
    ]
    best_val = float("inf")
    best_epoch = 0
    best_path = run_dir / "checkpoints" / f"{args.model}_{args.run_id}_best.pt"
    peak_gb = 0.0
    start = time.time()
    with (run_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, f"{args.model} {args.run_id} e{epoch:03d} train", not args.no_progress, args.log_interval, grad_clip)
            val_metrics = run_epoch(model, val_loader, criterion, None, device, f"{args.model} {args.run_id} e{epoch:03d} val", not args.no_progress, args.log_interval)
            if not all(math.isfinite(float(val_metrics[k])) for k in ("mse", "rmse", "mae")):
                raise NonFiniteTrainingError(phase=f"{args.model} {args.run_id} e{epoch:03d} val metrics", batch_idx=0)
            scheduler.step()
            epoch_s = time.time() - epoch_start
            gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
            peak_gb = max(peak_gb, gpu_peak)
            writer.writerow({
                "epoch": epoch,
                "train_mse": train_metrics["mse"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "train_r2": train_metrics["r2"],
                "val_mse": val_metrics["mse"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "val_pfa_rmse_g": val_metrics["pfa_rmse_g"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_max_grad_norm": train_metrics.get("max_grad_norm", 0.0),
                "epoch_s": epoch_s,
                "gpu_peak_gb": gpu_peak,
            })
            f.flush()
            print(f"{args.model} {args.run_id} epoch {epoch:03d}/{args.epochs} train_mse={train_metrics['mse']:.6f} val_mse={val_metrics['mse']:.6f} val_r2={val_metrics['r2']:.4f}")
            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                best_epoch = epoch
                save_checkpoint(best_path, model, optimizer, epoch, val_metrics, config)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

    summary = {
        "model": args.model,
        "run_id": args.run_id,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "train_time_s": time.time() - start,
        "peak_gpu_gb": peak_gb,
        "best_checkpoint": str(best_path),
        "params": stats,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    best_model = load_best_model(args.model, best_path, device)
    overall = evaluate_model(best_model, test_loader, device)
    overall_row = {"model": args.model, "run_id": args.run_id, "checkpoint": str(best_path), **summary, **stats, **overall}
    write_csv(eval_path, [overall_row])
    (run_dir / "evaluation_overall.json").write_text(json.dumps([overall_row], indent=2), encoding="utf-8")

    by_scale_rows = []
    mapping_by_id = {int(row["scale_id"]): row for row in mapping}
    for scale_id in range(57):
        scale_indices = flatten(split.test_gms, np.asarray([scale_id]))
        scale_ds = DynamicDataset(str(gm_path), str(building_dir), gm_indices=scale_indices)
        scale_loader = DataLoader(scale_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
        metrics = evaluate_model(best_model, scale_loader, device)
        m = mapping_by_id[scale_id]
        by_scale_rows.append({
            "model": args.model,
            "run_id": args.run_id,
            "scale_id": scale_id,
            "median_pga_ms2": m["median_pga_ms2"],
            "mean_pga_ms2": m["mean_pga_ms2"],
            "in_train_pga_range": m["in_train_pga_range"],
            **metrics,
        })
        print(f"{args.model} {args.run_id} scale={scale_id:02d} pga={float(m['median_pga_ms2']):.2f} mse={metrics['mse']:.6f} r2={metrics['r2']:.4f}")
    write_csv(run_dir / "evaluation_by_scale.csv", by_scale_rows)
    (run_dir / "evaluation_by_scale.json").write_text(json.dumps(by_scale_rows, indent=2), encoding="utf-8")
    print(f"Wrote O3 PGA OOD run to {run_dir}")


if __name__ == "__main__":
    main()
