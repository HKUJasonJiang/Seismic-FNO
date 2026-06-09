"""Train/evaluate O3 sampling-resolution OOD runs.

Train/validation are performed at a selected sampling rate while the fixed
474-GM x 57-scale test grid is always evaluated at the native 50Hz/3000-point
resolution.
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
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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

from baseline_models import PatchTransformer, build_model, count_parameters  # noqa: E402
from dataprep_v2 import DynamicDataset  # type: ignore  # noqa: E402
from evaluate_baselines import evaluate_model  # noqa: E402
from splits import make_fixed_474_split, save_split  # noqa: E402
from train_baselines import NonFiniteTrainingError, run_epoch, save_checkpoint, set_seed  # noqa: E402


DEFAULT_BASE_DATA = Path("/data/home/jason/data/SesimicTransformerData")
SAMPLE_HZ_TO_RUN_ID = {20: "S-20Hz", 25: "S-25Hz", 50: "S-50Hz", 10: "S-10Hz"}


class SamplingDataset(Dataset):
    """Wrap DynamicDataset and resample GM/response time histories."""

    def __init__(self, base_dataset: Dataset, target_hz: int, source_hz: int = 50):
        self.base_dataset = base_dataset
        self.target_hz = int(target_hz)
        self.source_hz = int(source_hz)
        self.target_len = int(round(3000 * self.target_hz / self.source_hz))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _resample_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.target_hz == self.source_hz:
            return tensor
        values = tensor.squeeze(-1).detach().cpu().numpy()
        if self.target_hz == 25:
            out = resample_poly(values, up=1, down=2)
        elif self.target_hz == 10:
            out = resample_poly(values, up=1, down=5)
        elif self.target_hz == 20:
            out = resample_poly(values, up=2, down=5)
        else:
            raise ValueError(f"Unsupported target_hz={self.target_hz}")
        if len(out) > self.target_len:
            out = out[: self.target_len]
        elif len(out) < self.target_len:
            out = np.pad(out, (0, self.target_len - len(out)))
        return torch.from_numpy(out.astype(np.float32, copy=False)).unsqueeze(-1)

    def __getitem__(self, idx: int):
        gm_data, attrs, acc_response, damage = self.base_dataset[idx]
        return self._resample_tensor(gm_data), attrs, self._resample_tensor(acc_response), damage


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


def build_sampling_model(model_name: str, seq_len: int) -> torch.nn.Module:
    if model_name == "FNO-Large":
        return build_fno_large()
    if model_name == "Transformer":
        return build_model("Transformer", seq_len=seq_len)
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


def infer_scale_mapping(gm_path: Path) -> list[dict[str, float | int]]:
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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def interpolate_pos_embed(pos_embed: torch.Tensor, target_patches: int) -> torch.Tensor:
    if pos_embed.shape[1] == target_patches:
        return pos_embed
    src = pos_embed.permute(0, 2, 1)
    dst = F.interpolate(src, size=target_patches, mode="linear", align_corners=True)
    return dst.permute(0, 2, 1)


def load_best_model(model_name: str, checkpoint_path: Path, device: str, eval_seq_len: int) -> torch.nn.Module:
    if model_name == "Transformer":
        model = PatchTransformer(seq_len=eval_seq_len)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = dict(checkpoint["model_state_dict"])
        target_patches = eval_seq_len // model.patch_size
        state["pos_embed"] = interpolate_pos_embed(state["pos_embed"], target_patches)
        model.load_state_dict(state, strict=True)
    else:
        model = build_sampling_model(model_name, eval_seq_len)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=(model_name != "FNO-Large"))
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate one O3 sampling-resolution OOD model.")
    parser.add_argument("--model", required=True, choices=["FNO-Large", "Transformer", "BiLSTM"])
    parser.add_argument("--train_hz", type=int, required=True, choices=[10, 20, 25, 50])
    parser.add_argument("--run_id", default=None)
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
    parser.add_argument("--output_root", default=str(O3_DIR / "results" / "sampling_ood"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    run_id = args.run_id or SAMPLE_HZ_TO_RUN_ID[args.train_hz]
    train_seq_len = int(round(3000 * args.train_hz / 50))
    eval_seq_len = 3000
    output_root = Path(args.output_root)
    run_dir = output_root / "runs" / args.model / run_id
    eval_path = run_dir / "evaluation_overall.csv"
    if eval_path.exists() and not args.force:
        print(f"Skipping {args.model}/{run_id}: {eval_path} exists.", flush=True)
        return

    base_data = Path(args.base_data_dir)
    gm_path = base_data / "MDOF" / "All_GMs" / "GMs_knet_3474_AF_57.h5"
    building_dir = Path(args.building_dir) if args.building_dir else base_data / "MDOF" / "knet-250" / "Data" / "fno"
    run_dir.mkdir(parents=True, exist_ok=True)

    mapping = infer_scale_mapping(gm_path)
    write_csv(run_dir / "scale_pga_mapping.csv", mapping)

    split = make_fixed_474_split(train_gms_used=3000, use_scales_count=57, seed=args.seed)
    train_indices = split.train_indices
    val_indices = split.val_indices
    test_indices = split.test_indices
    save_split(split, run_dir / "canonical_full_split_reference")

    train_base = DynamicDataset(str(gm_path), str(building_dir), gm_indices=train_indices)
    val_base = DynamicDataset(str(gm_path), str(building_dir), gm_indices=val_indices)
    test_ds = DynamicDataset(str(gm_path), str(building_dir), gm_indices=test_indices)
    train_ds = SamplingDataset(train_base, target_hz=args.train_hz)
    val_ds = SamplingDataset(val_base, target_hz=args.train_hz)

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_sampling_model(args.model, train_seq_len).to(device)
    stats = count_parameters(model)
    lr = model_lr(args.model, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = torch.nn.MSELoss()
    grad_clip = args.grad_clip if args.model in {"BiLSTM", "Transformer"} else 0.0

    config = {
        "model": args.model,
        "run_id": run_id,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": lr,
        "weight_decay": args.weight_decay,
        "scheduler_step": args.scheduler_step,
        "scheduler_gamma": args.scheduler_gamma,
        "grad_clip": grad_clip,
        "train_hz": args.train_hz,
        "train_seq_len": train_seq_len,
        "eval_hz": 50,
        "eval_seq_len": eval_seq_len,
        "train_rfft_bins": train_seq_len // 2 + 1,
        "eval_rfft_bins": eval_seq_len // 2 + 1,
        "fno_n_modes": 512 if args.model == "FNO-Large" else None,
        "resampling": "scipy.signal.resample_poly for train/val; fixed test remains native 50Hz",
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
    best_path = run_dir / "checkpoints" / f"{args.model}_{run_id}_best.pt"
    peak_gb = 0.0
    start = time.time()
    try:
        with (run_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
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
                    f"{args.model} {run_id} e{epoch:03d} train",
                    not args.no_progress,
                    args.log_interval,
                    grad_clip,
                )
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    None,
                    device,
                    f"{args.model} {run_id} e{epoch:03d} val",
                    not args.no_progress,
                    args.log_interval,
                )
                if not all(math.isfinite(float(val_metrics[k])) for k in ("mse", "rmse", "mae")):
                    raise NonFiniteTrainingError(phase=f"{args.model} {run_id} e{epoch:03d} val metrics", batch_idx=0)
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
                print(f"{args.model} {run_id} epoch {epoch:03d}/{args.epochs} train_mse={train_metrics['mse']:.6f} val_mse={val_metrics['mse']:.6f} val_r2={val_metrics['r2']:.4f}", flush=True)
                if val_metrics["mse"] < best_val:
                    best_val = val_metrics["mse"]
                    best_epoch = epoch
                    save_checkpoint(best_path, model, optimizer, epoch, val_metrics, config)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
    except NonFiniteTrainingError as exc:
        report = {"model": args.model, "run_id": run_id, "message": str(exc), "phase": exc.phase, "batch_idx": exc.batch_idx}
        (run_dir / "nonfinite_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise

    summary = {
        "model": args.model,
        "run_id": run_id,
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
    best_model = load_best_model(args.model, best_path, device, eval_seq_len)
    overall = evaluate_model(best_model, test_loader, device)
    overall_row = {"model": args.model, "run_id": run_id, "checkpoint": str(best_path), **summary, **stats, **overall}
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
            "run_id": run_id,
            "scale_id": scale_id,
            "median_pga_ms2": m["median_pga_ms2"],
            "mean_pga_ms2": m["mean_pga_ms2"],
            **metrics,
        })
        print(f"{args.model} {run_id} scale={scale_id:02d} pga={float(m['median_pga_ms2']):.2f} mse={metrics['mse']:.6f} r2={metrics['r2']:.4f}", flush=True)
    write_csv(run_dir / "evaluation_by_scale.csv", by_scale_rows)
    (run_dir / "evaluation_by_scale.json").write_text(json.dumps(by_scale_rows, indent=2), encoding="utf-8")
    print(f"Wrote O3 sampling OOD run to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
