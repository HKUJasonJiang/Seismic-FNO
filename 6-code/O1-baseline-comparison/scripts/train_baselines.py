"""Train O1 baseline models on the canonical full 57-factor split."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
PROJECT_MODULE_DIR = PROJECT_ROOT / "module"
UTILS_DIR = CODE_DIR / "shared" / "utils"
for path in (SCRIPT_DIR, UPSTREAM_DIR, UPSTREAM_MODULE_DIR, PROJECT_ROOT, PROJECT_MODULE_DIR, UTILS_DIR):
    if path.exists():
        sys.path.insert(0, str(path))

from baseline_models import BASELINE_REGISTRY, build_model, count_parameters  # noqa: E402
from dataprep_v2 import DynamicDataset  # type: ignore  # noqa: E402
from metrics import PFAMeter, RegressionMeter  # noqa: E402
from splits import make_fixed_474_split, save_split  # noqa: E402


DEFAULT_BASE_DATA = Path(r"D:\BaiduNetdiskDownload\SesimicTransformerData")


class NonFiniteTrainingError(RuntimeError):
    def __init__(self, *, phase: str, batch_idx: int, loss: float | None = None, grad_norm: float | None = None):
        self.phase = phase
        self.batch_idx = batch_idx
        self.loss = loss
        self.grad_norm = grad_norm
        parts = [f"Non-finite value detected during {phase}", f"batch={batch_idx}"]
        if loss is not None:
            parts.append(f"loss={loss}")
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm}")
        super().__init__(", ".join(parts))


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
    grad_clip: float | None = None,
) -> dict:
    train_mode = optimizer is not None
    model.train(train_mode)
    reg = RegressionMeter()
    pfa = PFAMeter()
    loss_sum = 0.0
    n_batches = 0
    max_grad_norm = 0.0
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
            if not torch.isfinite(loss).item():
                raise NonFiniteTrainingError(phase=desc, batch_idx=batch_idx, loss=float(loss.detach().cpu().item()))
            if train_mode:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip,
                        error_if_nonfinite=False,
                    )
                    grad_norm_value = float(grad_norm.detach().cpu().item())
                    if not math.isfinite(grad_norm_value):
                        raise NonFiniteTrainingError(
                            phase=desc,
                            batch_idx=batch_idx,
                            loss=float(loss.detach().cpu().item()),
                            grad_norm=grad_norm_value,
                        )
                    max_grad_norm = max(max_grad_norm, grad_norm_value)
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
    if train_mode:
        metrics["max_grad_norm"] = max_grad_norm
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


def write_nonfinite_report(path: Path, model_name: str, exc: NonFiniteTrainingError, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": model_name,
        "message": str(exc),
        "phase": exc.phase,
        "batch_idx": exc.batch_idx,
        "loss": exc.loss,
        "grad_norm": exc.grad_norm,
        "lr": effective_lr_for_model(model_name, args),
        "base_lr": args.lr,
        "bilstm_lr": args.bilstm_lr,
        "transformer_lr": args.transformer_lr,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def effective_lr_for_model(model_name: str, args) -> float:
    if model_name == "BiLSTM":
        return args.bilstm_lr
    if model_name == "Transformer":
        return args.transformer_lr
    return args.lr


def train_model(model_name: str, args, train_loader, val_loader) -> dict:
    set_seed(args.seed)
    device = args.device
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(model_name).to(device)
    stats = count_parameters(model)
    effective_lr = effective_lr_for_model(model_name, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=args.weight_decay)
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
        "lr": effective_lr,
        "base_lr": args.lr,
        "bilstm_lr": args.bilstm_lr,
        "transformer_lr": args.transformer_lr,
        "weight_decay": args.weight_decay,
        "scheduler_step": args.scheduler_step,
        "scheduler_gamma": args.scheduler_gamma,
        "grad_clip": args.grad_clip,
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
            "train_max_grad_norm",
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
                grad_clip=args.grad_clip,
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
            if not all(math.isfinite(float(val_metrics[key])) for key in ("mse", "rmse", "mae")):
                raise NonFiniteTrainingError(phase=f"{model_name} e{epoch:03d} val metrics", batch_idx=0)
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
                    "train_max_grad_norm": train_metrics.get("max_grad_norm", 0.0),
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
    parser.add_argument("--bilstm_lr", type=float, default=3e-4)
    parser.add_argument("--transformer_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
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
        try:
            summaries.append(train_model(model_name, args, train_loader, val_loader))
        except NonFiniteTrainingError as exc:
            report_path = Path(args.output_dir) / model_name / "nonfinite_report.json"
            write_nonfinite_report(report_path, model_name, exc, args)
            print(f"{exc}. Report saved to {report_path}", flush=True)
            raise SystemExit(2) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Training summary saved to {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
