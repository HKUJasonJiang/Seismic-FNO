"""Sequential queue runner for O3 sampling-resolution OOD jobs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
O3_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = O3_DIR.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run O3 sampling OOD jobs sequentially on visible GPU.")
    parser.add_argument("--models", nargs="+", required=True, choices=["FNO-Large", "Transformer", "BiLSTM"])
    parser.add_argument("--train_hz", nargs="+", type=int, default=[20, 25, 50], choices=[10, 20, 25, 50])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--base_data_dir", default="/data/home/jason/data/SesimicTransformerData")
    parser.add_argument("--building_dir", default="/data/home/jason/data/SesimicTransformerData/MDOF/knet-250/Data/fno")
    parser.add_argument("--output_root", default=str(O3_DIR / "results" / "sampling_ood"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    for model in args.models:
        for hz in args.train_hz:
            cmd = [
                args.python,
                str(SCRIPT_DIR / "train_eval_sampling_ood.py"),
                "--model",
                model,
                "--train_hz",
                str(hz),
                "--device",
                "cuda",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--eval_batch_size",
                str(args.eval_batch_size),
                "--base_data_dir",
                args.base_data_dir,
                "--building_dir",
                args.building_dir,
                "--output_root",
                args.output_root,
                "--log_interval",
                str(args.log_interval),
            ]
            if args.force:
                cmd.append("--force")
            if args.no_progress:
                cmd.append("--no_progress")
            print(f"\n=== O3 sampling queue start: {model} S-{hz}Hz ===", flush=True)
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            print(f"=== O3 sampling queue done: {model} S-{hz}Hz ===\n", flush=True)


if __name__ == "__main__":
    main()
