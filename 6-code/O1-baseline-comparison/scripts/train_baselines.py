"""Train O1 baseline models on the canonical full 57-factor split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm should be installed in the project venv
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
O1_DIR = SCRIPT_DIR.parent
CODE_DIR = O1_DIR.parent
PROJECT_ROOT = CODE_DIR.parent
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
from splits import make_fixed_474_split, save_split  # noqa: E402


DEFAULT_BASE_DATA = Path(r"D:\BaiduNetdiskDownload\SesimicTransformerData")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def move_batch(batch, device: str):
    gm_data, _attrs, acc_response, _damage = batch
    x = gm_data.permute(0, 2, 1).to(device, non_blocking=True)
    y = acc_response.permute(0, 2, 1).to(device, non_blocking=True)
    return x, y


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str | None = None,
    desc: str = "",
    show_progress: bool = True,
    log_interval: int = 100,
) -> dict:
    train_mode = optimizer is not None
    model.train(train_mode)
    reg = RegressionMeter()
    pfa = PFAMeter()
    loss_sum = 0.0
    n_batches = 0
    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True)
    for batch_idx, batch in enumerate(iterator, start=1):
        x, y = move_batch(batch, device or "cpu")
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            pred = model(x)
            loss = criterion(pred, y)
            if train_mode:
                loss.backward()
                optimizer.step()
        reg.update(pred, y)
        pfa.update(pred, y)
        loss_sum += float(loss.item())
        n_batches += 1
        if show_progress and tqdm is not None:
            iterator.set_postfix(loss=f"{loss_sum / n_batches:.5f}", refresh=False)
        elif show_progress and log_interval > 0 and batch_idx % log_interval == 0:
            print(f"{desc} batch {batch_idx}/{len(loader)} loss={loss_sum / n_batches:.5f}", flush=True)
    metrics = reg.compute()
    metrics.update(pfa.compute())
    metrics["loss"] = loss_sum / max(n_batches, 1)
    return metrics


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def train_model(model_name: str, args, train_loader, val_loader) -> dict:
    set_seed(args.seed)
    device = args.device
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(model_name).to(device)
    stats = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = torch.nn.MSELoss()

    run_dir = Path(args.output_dir) / model_name
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
        "params": stats,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val = float("inf")
    best_path = run_dir / "checkpoints" / f"{model_name}_best.pt"
    start = time.time()
    peak_gb = 0.0
    with log_path.open("w", newline="", encoding="utf-8") as f:
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
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train O1 baselines.")
    parser.add_argument("--models", nargs="+", default=list(BASELINE_REGISTRY))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_step", type=int, default=20)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_data_dir", default=str(DEFAULT_BASE_DATA))
    parser.add_argument("--building_dir", default=None)
    parser.add_argument("--output_dir", default=str(O1_DIR / "results" / "runs"))
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm/batch progress output.")
    parser.add_argument("--log_interval", type=int, default=100, help="Fallback batch print interval if tqdm is unavailable.")
    args = parser.parse_args()

    set_seed(args.seed)
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
    pin = args.device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    summaries = []
    for model_name in args.models:
        if model_name not in BASELINE_REGISTRY:
            raise ValueError(f"Unknown model {model_name}. Options: {list(BASELINE_REGISTRY)}")
        summaries.append(train_model(model_name, args, train_loader, val_loader))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Training summary saved to {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
