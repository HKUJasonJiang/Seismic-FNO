"""VRAM preflight for O1 baseline comparison.

This script probes candidate batch sizes for every model before formal
training. It records CUDA peak allocated memory, which reflects dedicated GPU
memory used by PyTorch. Shared GPU memory is not treated as usable budget.

Example:
    python scripts/vram_preflight.py --device cuda --batches 32 64 128 256
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
O1_DIR = SCRIPT_DIR.parent
CODE_DIR = O1_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from baseline_models import BASELINE_REGISTRY, build_model, count_parameters  # noqa: E402


def run_probe(model_name: str, batch_size: int, seq_len: int, device: str) -> dict:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_model(model_name).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    x = torch.randn(batch_size, 1, seq_len, device=device)
    y = torch.randn(batch_size, 1, seq_len, device=device)

    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    elapsed = time.time() - start

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
    reserved_gb = torch.cuda.max_memory_reserved() / 1024**3 if device == "cuda" else 0.0
    del model, optimizer, x, y, pred, loss
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "batch_size": batch_size,
        "status": "ok",
        "peak_allocated_gb": peak_gb,
        "peak_reserved_gb": reserved_gb,
        "elapsed_s": elapsed,
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run O1 VRAM preflight.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=list(BASELINE_REGISTRY))
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--seq_len", type=int, default=3000)
    parser.add_argument("--output", default=str(O1_DIR / "results" / "vram_preflight.csv"))
    parser.add_argument("--max_fraction", type=float, default=0.90, help="Max fraction of dedicated VRAM allowed.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_gb = 0.0
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        print(f"GPU: {props.name}")
        print(f"Dedicated CUDA memory: {total_gb:.2f} GB")
        print(f"Budget at max_fraction={args.max_fraction}: {total_gb * args.max_fraction:.2f} GB")

    rows = []
    for model_name in args.models:
        stats = count_parameters(build_model(model_name))
        print(f"\n[{model_name}] params={stats['numel_m']:.2f}M")
        for batch_size in args.batches:
            try:
                row = run_probe(model_name, batch_size, args.seq_len, args.device)
                if total_gb and row["peak_allocated_gb"] > total_gb * args.max_fraction:
                    row["status"] = "over_budget"
                print(
                    f"  bs={batch_size:<5d} {row['status']:<11s} "
                    f"peak={row['peak_allocated_gb']:.2f}GB reserved={row['peak_reserved_gb']:.2f}GB"
                )
            except RuntimeError as exc:
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                row = {
                    "model": model_name,
                    "batch_size": batch_size,
                    "status": "oom_or_error",
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0.0,
                    "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if args.device == "cuda" else 0.0,
                    "elapsed_s": 0.0,
                    "error": str(exc).replace("\n", " ")[:500],
                }
                print(f"  bs={batch_size:<5d} oom_or_error")
            rows.append(row)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    viable_by_model = {}
    for model_name in args.models:
        viable = [
            row["batch_size"]
            for row in rows
            if row["model"] == model_name and row["status"] == "ok"
        ]
        viable_by_model[model_name] = max(viable) if viable else None

    if all(v is not None for v in viable_by_model.values()):
        selected = min(v for v in viable_by_model.values() if v is not None)
        print(f"\nRecommended unified batch size: {selected}")
    else:
        print("\nNo unified batch size found for all models in the probed candidates.")
    print(f"Preflight CSV saved to: {output}")


if __name__ == "__main__":
    main()

