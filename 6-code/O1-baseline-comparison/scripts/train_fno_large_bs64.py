"""Train FNO-Large under the O1 fair-comparison protocol.

This run is separate from the legacy high-throughput FNO-Large checkpoint.
Its purpose is a strict architectural comparison:

    FNO-Large vs. baselines
    same split, same 57 factors, same optimizer/scheduler/epochs, same batch size.

Example:
    python scripts/train_fno_large_bs64.py --device cuda --batch_size 64 --epochs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

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

from baseline_models import count_parameters  # noqa: E402
from dataprep_v2 import DynamicDataset  # type: ignore  # noqa: E402
from splits import make_fixed_474_split, save_split  # noqa: E402
from train_baselines import run_epoch, save_checkpoint, set_seed  # noqa: E402


DEFAULT_BASE_DATA = Path(r"D:\BaiduNetdiskDownload\SesimicTransformerData")


def build_fno_large() -> torch.nn.Module:
    try:
        from neuralop.models import FNO
    except ModuleNotFoundError as exc:
        if exc.name == "tltorch":
            raise ModuleNotFoundError(
                "FNO-Large requires the tensorly-torch package that provides `tltorch`. "
                "Install it in the active venv before running this script, e.g. "
                "`python -m pip install tensorly-torch`."
            ) from exc
        raise
    return FNO(
        n_modes=(512,),
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        projection_channel_ratio=2,
        n_layers=8,
        domain_padding=0.1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FNO-Large with O1 bs=64 fair protocol.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_step", type=int, default=20)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_data_dir", default=str(DEFAULT_BASE_DATA))
    parser.add_argument("--building_dir", default=None)
    parser.add_argument("--output_dir", default=str(O1_DIR / "results" / "runs" / "FNO-Large-bs64"))
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm/batch progress output.")
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    base_data = Path(args.base_data_dir)
    gm_path = base_data / "MDOF" / "All_GMs" / "GMs_knet_3474_AF_57.h5"
    building_dir = Path(args.building_dir) if args.building_dir else base_data / "MDOF" / "knet-250" / "Data" / "fno"

    split = make_fixed_474_split(use_scales_count=57, seed=args.seed)
    save_split(split, CODE_DIR / "shared" / "split_indices")
    print(f"Train samples={len(split.train_indices):,}, Val={len(split.val_indices):,}, Test={len(split.test_indices):,}")
    print(f"GM path: {gm_path}")
    print(f"Building dir: {building_dir}")

    train_ds = DynamicDataset(str(gm_path), str(building_dir), gm_indices=split.train_indices)
    val_ds = DynamicDataset(str(gm_path), str(building_dir), gm_indices=split.val_indices)
    pin = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model_name = "FNO-Large-bs64"
    model = build_fno_large().to(device)
    stats = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = torch.nn.MSELoss()

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training_log.csv"
    config = {
        "model": model_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler_step": args.scheduler_step,
        "scheduler_gamma": args.scheduler_gamma,
        "split": "fixed_474_test_full_57_factors",
        "architecture": {
            "n_modes": 512,
            "hidden_channels": 64,
            "n_layers": 8,
            "domain_padding": 0.1,
        },
        "params": stats,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    import csv

    fieldnames = [
        "epoch",
        "train_mse",
        "train_rmse",
        "train_mae",
        "train_r2",
        "val_mse",
        "val_rmse",
        "val_mae",
        "val_r2",
        "val_pfa_rmse_g",
        "lr",
        "epoch_s",
        "gpu_peak_gb",
    ]
    best_val = float("inf")
    best_path = run_dir / "checkpoints" / f"{model_name}_best.pt"
    peak_gb = 0.0
    start = time.time()
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                desc=f"{model_name} e{epoch:03d} train",
                show_progress=not args.no_progress,
                log_interval=args.log_interval,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                device,
                desc=f"{model_name} e{epoch:03d} val",
                show_progress=not args.no_progress,
                log_interval=args.log_interval,
            )
            scheduler.step()
            epoch_s = time.time() - epoch_start
            gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
            peak_gb = max(peak_gb, gpu_peak)
            writer.writerow(
                {
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
                    "epoch_s": epoch_s,
                    "gpu_peak_gb": gpu_peak,
                }
            )
            f.flush()
            print(
                f"{model_name} epoch {epoch:03d}/{args.epochs} "
                f"train_mse={train_metrics['mse']:.6f} val_mse={val_metrics['mse']:.6f} "
                f"val_r2={val_metrics['r2']:.4f} peak={gpu_peak:.2f}GB"
            )
            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                save_checkpoint(best_path, model, optimizer, epoch, val_metrics, config)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

    summary = {
        "model": model_name,
        "best_val_mse": best_val,
        "train_time_s": time.time() - start,
        "peak_gpu_gb": peak_gb,
        "best_checkpoint": str(best_path),
        "params": stats,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
