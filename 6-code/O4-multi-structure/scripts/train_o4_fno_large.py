"""Train FNO-Large for the O4 six-structure validation package.

This script intentionally accepts explicit single-building HDF5 files through
`configs/o4_structures.json`. It avoids the old implicit `Data/fno` workflow
where every `.h5` in a directory silently becomes part of the dataset.

Example:
    python scripts/train_o4_fno_large.py --structures all --device cuda --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
O4_DIR = SCRIPT_DIR.parent
CODE_DIR = O4_DIR.parent
UPSTREAM_DIR = CODE_DIR / "shared" / "upstream" / "Seismic-FNO-clean"
UTILS_DIR = CODE_DIR / "shared" / "utils"
sys.path.insert(0, str(UPSTREAM_DIR))
sys.path.insert(0, str(UTILS_DIR))

from metrics import PFAMeter, RegressionMeter  # noqa: E402
from splits import make_fixed_474_split, save_split  # noqa: E402


DEFAULT_BASE_DATA = Path(r"D:\BaiduNetdiskDownload\SesimicTransformerData")
STRUCTURE_CONFIG = O4_DIR / "configs" / "o4_structures.json"


class SingleBuildingDataset(Dataset):
    """Lazy HDF5 dataset for one structure and selected global GM indices."""

    def __init__(self, gm_h5: str | Path, building_h5: str | Path, indices: np.ndarray):
        self.gm_h5 = str(gm_h5)
        self.building_h5 = str(building_h5)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.gm_file = None
        self.building_file = None

    def __len__(self) -> int:
        return len(self.indices)

    def _open(self) -> None:
        self.gm_file = h5py.File(self.gm_h5, "r")
        self.building_file = h5py.File(self.building_h5, "r")

    def __getitem__(self, item: int):
        if self.gm_file is None or self.building_file is None:
            self._open()
        idx = int(self.indices[item])
        gm = torch.from_numpy(self.gm_file["Acc_GMs"][idx]).float().unsqueeze(-1)
        response = torch.from_numpy(self.building_file["Acc_Floor_Response"][idx]).float().unsqueeze(-1)
        damage = torch.from_numpy(self.building_file["Blg_Damage_State"][idx]).long()
        return gm, {}, response, damage


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_fno_large() -> torch.nn.Module:
    try:
        from neuralop.models import FNO
    except ModuleNotFoundError as exc:
        if exc.name == "tltorch":
            raise ModuleNotFoundError(
                "FNO-Large requires tensorly-torch (`tltorch`). "
                "Install it first: python -m pip install tensorly-torch"
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


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def move_batch(batch, device: str):
    gm_data, _attrs, acc_response, _damage = batch
    x = gm_data.permute(0, 2, 1).to(device, non_blocking=True)
    y = acc_response.permute(0, 2, 1).to(device, non_blocking=True)
    return x, y


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: str,
    desc: str,
    show_progress: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    reg = RegressionMeter()
    pfa = PFAMeter()
    loss_sum = 0.0
    n_batches = 0
    iterator: Iterable = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True)
    for batch_idx, batch in enumerate(iterator, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        x, y = move_batch(batch, device)
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
            iterator.set_postfix(loss=f"{loss_sum / max(n_batches, 1):.5f}", refresh=False)
    metrics = reg.compute()
    metrics.update(pfa.compute())
    metrics["loss"] = loss_sum / max(n_batches, 1)
    return metrics


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, config: dict, scheduler=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def log_line(message: str, log_file=None) -> None:
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def save_training_curves(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ax = axes[0]
    ax.plot(epochs, [row["train_loss"] for row in rows], label="train")
    ax.plot(epochs, [row["val_loss"] for row in rows], label="val")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSELoss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, [row["train_mse"] for row in rows], label="train")
    ax.plot(epochs, [row["val_mse"] for row in rows], label="val")
    ax.set_title("Response MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(epochs, [row["train_pfa_mse_g2"] for row in rows], label="train")
    ax.plot(epochs, [row["val_pfa_mse_g2"] for row in rows], label="val")
    ax.set_title("PFA MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("g^2")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_structures() -> dict:
    return json.loads(STRUCTURE_CONFIG.read_text(encoding="utf-8"))


def resolve_structure_names(requested: list[str], structures: dict) -> list[str]:
    if requested == ["all"]:
        return list(structures)
    bad = [name for name in requested if name not in structures]
    if bad:
        raise ValueError(f"Unknown structures {bad}. Options: {list(structures)} or all")
    return requested


def train_one_structure(name: str, meta: dict, split, args) -> dict:
    set_seed(args.seed)
    device = args.device
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    data_dir = Path(args.data_dir)
    gm_path = Path(args.gm_path) if args.gm_path else Path(args.base_data_dir) / "MDOF" / "All_GMs" / "GMs_knet_3474_AF_57.h5"
    building_path = data_dir / meta["filename"]
    if not gm_path.exists():
        raise FileNotFoundError(f"GM file not found: {gm_path}")
    if not building_path.exists():
        raise FileNotFoundError(f"Building HDF5 not found for {name}: {building_path}")

    train_ds = SingleBuildingDataset(gm_path, building_path, split.train_indices)
    val_ds = SingleBuildingDataset(gm_path, building_path, split.val_indices)
    test_ds = SingleBuildingDataset(gm_path, building_path, split.test_indices)
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model_name = f"FNO-Large_{name}"
    model = build_fno_large().to(device)
    if args.torch_compile:
        model = torch.compile(model)
    stats = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = torch.nn.MSELoss()

    run_dir = Path(args.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "objective": "O4-multi-structure",
        "structure_id": name,
        "structure": meta,
        "gm_path": str(gm_path),
        "building_h5": str(building_path),
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
            "projection_channel_ratio": 2,
        },
        "params": stats,
        "artifacts": {
            "training_log_csv": str(run_dir / "training_log.csv"),
            "training_console_log": str(run_dir / "training_console.log"),
            "training_curves_png": str(run_dir / "figures" / "training_curves.png"),
            "test_metrics_json": str(run_dir / "test_metrics.json"),
            "last_checkpoint": str(run_dir / "checkpoints" / f"FNO-Large_{name}_last.pt"),
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log_path = run_dir / "training_log.csv"
    console_log_path = run_dir / "training_console.log"
    curves_path = run_dir / "figures" / "training_curves.png"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_mse",
        "train_pfa_mse_g2",
        "val_loss",
        "val_mse",
        "val_pfa_mse_g2",
        "lr",
        "epoch_s",
        "gpu_peak_gb",
    ]
    best_val = float("inf")
    best_epoch = 0
    best_path = run_dir / "checkpoints" / f"{model_name}_best.pt"
    last_path = run_dir / "checkpoints" / f"{model_name}_last.pt"
    peak_gb = 0.0
    start = time.time()
    curve_rows = []

    with console_log_path.open("w", encoding="utf-8") as console_log, log_path.open("w", newline="", encoding="utf-8") as f:
        log_line(f"\n=== O4 {name} / {meta['paper_label']} ===", console_log)
        log_line(f"Building: {building_path}", console_log)
        log_line(f"GM: {gm_path}", console_log)
        log_line(f"Train={len(split.train_indices):,}, Val={len(split.val_indices):,}, Test={len(split.test_indices):,}", console_log)
        log_line(f"Parameters: total={stats['total']:,}, trainable={stats['trainable']:,}", console_log)
        log_line(f"CSV log: {log_path}", console_log)
        log_line(f"Curve PNG: {curves_path}", console_log)

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
                desc=f"{name} e{epoch:03d} train",
                show_progress=not args.no_progress,
                max_batches=args.max_batches_per_epoch,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                device,
                desc=f"{name} e{epoch:03d} val",
                show_progress=not args.no_progress,
                max_batches=args.max_batches_per_epoch,
            )
            scheduler.step()
            epoch_s = time.time() - epoch_start
            gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
            peak_gb = max(peak_gb, gpu_peak)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "train_pfa_mse_g2": train_metrics["pfa_mse_g2"],
                "val_loss": val_metrics["loss"],
                "val_mse": val_metrics["mse"],
                "val_pfa_mse_g2": val_metrics["pfa_mse_g2"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_s": epoch_s,
                "gpu_peak_gb": gpu_peak,
            }
            writer.writerow(row)
            f.flush()
            curve_rows.append(row)
            save_training_curves(curve_rows, curves_path)
            log_line(
                f"{name} epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
                f"train_mse={train_metrics['mse']:.6f} val_mse={val_metrics['mse']:.6f} "
                f"val_pfa_mse={val_metrics['pfa_mse_g2']:.6f}g^2 "
                f"peak={gpu_peak:.2f}GB",
                console_log,
            )
            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                best_epoch = epoch
                save_checkpoint(best_path, model, optimizer, epoch, val_metrics, config, scheduler=scheduler)
            save_checkpoint(last_path, model, optimizer, epoch, val_metrics, config, scheduler=scheduler)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        None,
        device,
        desc=f"{name} test",
        show_progress=not args.no_progress,
        max_batches=args.max_test_batches,
    )
    test_metrics_path = run_dir / "test_metrics.json"
    test_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    with console_log_path.open("a", encoding="utf-8") as console_log:
        log_line(
            f"{name} final test: test_mse={test_metrics['mse']:.6f} "
            f"test_pfa_mse={test_metrics['pfa_mse_g2']:.6f}g^2 "
            f"best_epoch={best_epoch}",
            console_log,
        )
    summary = {
        "structure_id": name,
        "paper_label": meta["paper_label"],
        "type": meta["type"],
        "stories": meta["stories"],
        "height_m": meta["height_m"],
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "test_mse": test_metrics["mse"],
        "test_pfa_mse_g2": test_metrics["pfa_mse_g2"],
        "test_metrics": test_metrics,
        "train_time_s": time.time() - start,
        "peak_gpu_gb": peak_gb,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "config": str(run_dir / "config.json"),
        "training_log_csv": str(log_path),
        "training_console_log": str(console_log_path),
        "training_curves_png": str(curves_path),
        "test_metrics_json": str(test_metrics_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{name} summary saved to {run_dir / 'summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train O4 FNO-Large on six target structures.")
    parser.add_argument("--structures", nargs="+", default=["all"], help="Structure IDs or all.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_step", type=int, default=20)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_data_dir", default=str(DEFAULT_BASE_DATA))
    parser.add_argument("--data_dir", default=str(DEFAULT_BASE_DATA / "MDOF" / "knet-250" / "Data"))
    parser.add_argument("--gm_path", default=None)
    parser.add_argument("--output_dir", default=str(O4_DIR / "results" / "runs"))
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--max_batches_per_epoch", type=int, default=None, help="Debug/smoke limit.")
    parser.add_argument("--max_test_batches", type=int, default=None, help="Debug/smoke test limit.")
    args = parser.parse_args()

    structures = load_structures()
    selected = resolve_structure_names(args.structures, structures)
    split = make_fixed_474_split(use_scales_count=57, seed=args.seed)
    save_split(split, CODE_DIR / "shared" / "split_indices")

    all_summaries = []
    for name in selected:
        all_summaries.append(train_one_structure(name, structures[name], split, args))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "o4_training_summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"O4 summary saved to {summary_path}")


if __name__ == "__main__":
    main()
