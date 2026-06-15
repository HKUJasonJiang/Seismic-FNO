"""Train and evaluate one O2 data-efficiency run.

The O2 protocol keeps the fixed 474-GM test set unchanged and only prunes the
supervised train/validation pool by GM count and/or AC scale count.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
O2_DIR = SCRIPT_DIR.parent
CODE_DIR = O2_DIR.parent
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

EFFICIENCY_CONFIGS: dict[str, dict[str, int | str]] = {
    "E-Base": {"gms": 3000, "scales": 57, "axis": "full"},
    "E-AC80": {"gms": 3000, "scales": 46, "axis": "ac"},
    "E-AC60": {"gms": 3000, "scales": 34, "axis": "ac"},
    "E-AC40": {"gms": 3000, "scales": 23, "axis": "ac"},
    "E-AC20": {"gms": 3000, "scales": 11, "axis": "ac"},
    "E-AC10": {"gms": 3000, "scales": 6, "axis": "ac"},
    "E-AC05": {"gms": 3000, "scales": 3, "axis": "ac"},
    "E-GM80": {"gms": 2400, "scales": 57, "axis": "gm"},
    "E-GM60": {"gms": 1800, "scales": 57, "axis": "gm"},
    "E-GM40": {"gms": 1200, "scales": 57, "axis": "gm"},
    "E-GM20": {"gms": 600, "scales": 57, "axis": "gm"},
    "E-GM10": {"gms": 300, "scales": 57, "axis": "gm"},
    "E-GM05": {"gms": 150, "scales": 57, "axis": "gm"},
    "E-J80": {"gms": 2400, "scales": 46, "axis": "joint"},
    "E-J60": {"gms": 1800, "scales": 34, "axis": "joint"},
    "E-J40": {"gms": 1200, "scales": 23, "axis": "joint"},
    "E-J20": {"gms": 600, "scales": 11, "axis": "joint"},
    "E-J10": {"gms": 300, "scales": 6, "axis": "joint"},
    "E-J05": {"gms": 150, "scales": 3, "axis": "joint"},
}

REDUCED_RUN_IDS = [run_id for run_id in EFFICIENCY_CONFIGS if run_id != "E-Base"]


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


def model_lr(model_name: str, args: argparse.Namespace) -> float:
    if model_name == "BiLSTM":
        return args.bilstm_lr
    if model_name == "Transformer":
        return args.transformer_lr
    return args.lr


def build_o2_model(model_name: str) -> torch.nn.Module:
    if model_name == "FNO-Large":
        return build_fno_large()
    return build_model(model_name)


def resolve_single_structure_path(path: Path) -> Path:
    """Return one HDF5 response file, failing loudly for ambiguous directories."""
    if path.is_file():
        if path.suffix != ".h5":
            raise ValueError(f"Building file must be an .h5 file: {path}")
        return path

    h5_files = sorted(path.glob("*.h5"))
    if len(h5_files) != 1:
        names = ", ".join(p.name for p in h5_files) or "none"
        raise ValueError(
            "This training script uses models that receive only ground motion as input. "
            f"Expected exactly one building response .h5, but found {len(h5_files)} in {path}: {names}. "
            "Pass --building_dir as a single .h5 file for the target structure."
        )
    return h5_files[0]


def checkpoint_model_name(model_name: str) -> str:
    return "FNO-Large" if model_name == "FNO-Large" else model_name


def load_best_model(model_name: str, checkpoint_path: Path, device: str) -> torch.nn.Module:
    model = build_o2_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=(model_name != "FNO-Large"))
    return model.to(device)


def train_one(args: argparse.Namespace, run_id: str, run_cfg: dict[str, int | str]) -> dict:
    set_seed(args.seed)
    device = args.device
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    output_root = Path(args.output_root)
    run_dir = output_root / "runs" / args.model / run_id
    eval_path = run_dir / "evaluation.csv"
    if eval_path.exists() and not args.force:
        print(f"Skipping {args.model}/{run_id}: evaluation exists at {eval_path}", flush=True)
        return json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))[0]

    run_dir.mkdir(parents=True, exist_ok=True)
    base_data = Path(args.base_data_dir)
    gm_path = base_data / "MDOF" / "All_GMs" / "GMs_knet_3474_AF_57.h5"
    building_dir = Path(args.building_dir) if args.building_dir else base_data / "MDOF" / "knet-250" / "Data" / "fno"
    building_path = resolve_single_structure_path(building_dir)

    train_gms_used = int(run_cfg["gms"])
    use_scales_count = int(run_cfg["scales"])
    split = make_fixed_474_split(
        train_gms_used=train_gms_used,
        use_scales_count=use_scales_count,
        seed=args.seed,
    )
    save_split(split, run_dir)

    train_ds = DynamicDataset(str(gm_path), str(building_path), gm_indices=split.train_indices)
    val_ds = DynamicDataset(str(gm_path), str(building_path), gm_indices=split.val_indices)
    test_ds = DynamicDataset(str(gm_path), str(building_path), gm_indices=split.test_indices)
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model = build_o2_model(args.model).to(device)
    stats = count_parameters(model)
    lr = model_lr(args.model, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = torch.nn.MSELoss()
    grad_clip = args.grad_clip if args.model in {"BiLSTM", "Transformer"} else 0.0

    config = {
        "model": args.model,
        "run_id": run_id,
        "axis": run_cfg["axis"],
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": lr,
        "base_lr": args.lr,
        "bilstm_lr": args.bilstm_lr,
        "transformer_lr": args.transformer_lr,
        "weight_decay": args.weight_decay,
        "scheduler_step": args.scheduler_step,
        "scheduler_gamma": args.scheduler_gamma,
        "grad_clip": grad_clip,
        "train_gms_used": train_gms_used,
        "use_scales_count": use_scales_count,
        "train_gms": len(split.train_gms),
        "val_gms": len(split.val_gms),
        "test_gms": len(split.test_gms),
        "train_samples": len(split.train_indices),
        "val_samples": len(split.val_indices),
        "test_samples": len(split.test_indices),
        "building_file": str(building_path),
        "scale_indices": split.scale_indices.tolist(),
        "split": "fixed_474_test_full_57_factors_reduced_train_val",
        "params": stats,
    }
    if args.model == "FNO-Large":
        config["architecture"] = {
            "n_modes": 512,
            "hidden_channels": 64,
            "n_layers": 8,
            "domain_padding": 0.1,
        }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

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
    best_val = float("inf")
    best_epoch = 0
    best_path = run_dir / "checkpoints" / f"{checkpoint_model_name(args.model)}_{run_id}_best.pt"
    best_state = None
    peak_gb = 0.0
    start = time.time()
    log_path = run_dir / "training_log.csv"
    try:
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
                    desc=f"{args.model} {run_id} e{epoch:03d} train",
                    show_progress=not args.no_progress,
                    log_interval=args.log_interval,
                    grad_clip=grad_clip,
                )
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    None,
                    device,
                    desc=f"{args.model} {run_id} e{epoch:03d} val",
                    show_progress=not args.no_progress,
                    log_interval=args.log_interval,
                )
                if not all(math.isfinite(float(val_metrics[key])) for key in ("mse", "rmse", "mae")):
                    raise NonFiniteTrainingError(phase=f"{args.model} {run_id} e{epoch:03d} val metrics", batch_idx=0)
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
                    f"{args.model} {run_id} epoch {epoch:03d}/{args.epochs} "
                    f"train_mse={train_metrics['mse']:.6f} val_mse={val_metrics['mse']:.6f} "
                    f"val_r2={val_metrics['r2']:.4f} peak={gpu_peak:.2f}GB",
                    flush=True,
                )
                if val_metrics["mse"] < best_val:
                    best_val = val_metrics["mse"]
                    best_epoch = epoch
                    best_state = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in model.state_dict().items()}
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
    except NonFiniteTrainingError as exc:
        report = {
            "model": args.model,
            "run_id": run_id,
            "message": str(exc),
            "phase": exc.phase,
            "batch_idx": exc.batch_idx,
            "loss": exc.loss,
            "grad_norm": exc.grad_norm,
            "config": config,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (run_dir / "nonfinite_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(2) from exc

    summary = {
        "model": args.model,
        "run_id": run_id,
        "axis": run_cfg["axis"],
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "train_time_s": time.time() - start,
        "peak_gpu_gb": peak_gb,
        "best_checkpoint": "",
        "params": stats,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if best_state is None:
        best_state = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in model.state_dict().items()}

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    best_model = build_o2_model(args.model).to(device)
    best_model.load_state_dict(best_state, strict=(args.model != "FNO-Large"))
    metrics = evaluate_model(best_model, test_loader, device)
    row = {
        "model": args.model,
        "run_id": run_id,
        "axis": run_cfg["axis"],
        "train_gms_used": train_gms_used,
        "use_scales_count": use_scales_count,
        "pool_samples": train_gms_used * use_scales_count,
        "train_samples": len(split.train_indices),
        "val_samples": len(split.val_indices),
        "test_samples": len(split.test_indices),
        "best_epoch": best_epoch,
        "train_time_s": summary["train_time_s"],
        "peak_gpu_gb": peak_gb,
        "checkpoint": "",
        **stats,
        **metrics,
    }
    with eval_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    (run_dir / "evaluation.json").write_text(json.dumps([row], indent=2), encoding="utf-8")
    print(f"{args.model} {run_id} test_mse={metrics['mse']:.6f} test_r2={metrics['r2']:.4f}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate one O2 data-efficiency run.")
    parser.add_argument("--model", required=True, choices=["FNO-Large", "Transformer", "BiLSTM"])
    parser.add_argument("--run_id", required=True, choices=list(EFFICIENCY_CONFIGS))
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
    parser.add_argument("--output_root", default=str(O2_DIR / "results"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()
    train_one(args, args.run_id, EFFICIENCY_CONFIGS[args.run_id])


if __name__ == "__main__":
    main()
