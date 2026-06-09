"""Run an O2 model queue sequentially on the visible GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from train_eval_o2 import REDUCED_RUN_IDS


SCRIPT_DIR = Path(__file__).resolve().parent
O2_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = O2_DIR.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential O2 queue runner.")
    parser.add_argument("--models", nargs="+", required=True, choices=["FNO-Large", "Transformer", "BiLSTM"])
    parser.add_argument("--run_ids", nargs="+", default=REDUCED_RUN_IDS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--base_data_dir", default="/data/home/jason/data/SesimicTransformerData")
    parser.add_argument("--building_dir", default="/data/home/jason/data/SesimicTransformerData/MDOF/knet-250/Data/fno")
    parser.add_argument("--output_root", default=str(O2_DIR / "results"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    for model in args.models:
        for run_id in args.run_ids:
            cmd = [
                args.python,
                str(SCRIPT_DIR / "train_eval_o2.py"),
                "--model",
                model,
                "--run_id",
                run_id,
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
            print("\n=== O2 queue start:", model, run_id, "===", flush=True)
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            print("=== O2 queue done:", model, run_id, "===\n", flush=True)


if __name__ == "__main__":
    main()
