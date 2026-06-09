"""Collect O2 data-efficiency results into a manuscript-style table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
O2_DIR = SCRIPT_DIR.parent
O1_DIR = O2_DIR.parent / "O1-baseline-comparison"


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def o1_base_rows(o1_eval: Path, o1_runs: Path) -> list[dict]:
    if not o1_eval.exists():
        return []
    eval_df = pd.read_csv(o1_eval)
    model_map = {
        "FNO-Large": "FNO-Large-bs64",
        "Transformer": "Transformer",
        "BiLSTM": "BiLSTM",
    }
    rows = []
    for model, run_dir_name in model_map.items():
        match = eval_df[eval_df["model"] == model]
        if match.empty:
            continue
        eval_row = match.iloc[0].to_dict()
        run_dir = o1_runs / run_dir_name
        config = load_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
        summary = load_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
        row = {
            "model": model,
            "run_id": "E-Base",
            "axis": "full",
            "train_gms_used": 3000,
            "use_scales_count": 57,
            "pool_samples": 171000,
            "train_samples": 136800,
            "val_samples": 34200,
            "test_samples": 27018,
            "best_epoch": summary.get("best_epoch", ""),
            "train_time_s": summary.get("train_time_s", ""),
            "peak_gpu_gb": summary.get("peak_gpu_gb", ""),
            "epochs": config.get("epochs", ""),
            "batch_size": config.get("batch_size", ""),
            "lr": config.get("lr", ""),
        }
        for key, value in eval_row.items():
            if key not in row:
                row[key] = value
        rows.append(row)
    return rows


def reduced_rows(results_root: Path) -> list[dict]:
    rows = []
    for eval_path in sorted((results_root / "runs").glob("*/*/evaluation.csv")):
        run_dir = eval_path.parent
        row = pd.read_csv(eval_path).iloc[0].to_dict()
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        config = load_json(config_path) if config_path.exists() else {}
        summary = load_json(summary_path) if summary_path.exists() else {}
        row["epochs"] = config.get("epochs", "")
        row["batch_size"] = config.get("batch_size", "")
        row["lr"] = config.get("lr", "")
        row["best_val_mse"] = summary.get("best_val_mse", "")
        rows.append(row)
    return rows


def add_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["r2_retention"] = float("nan")
    df["mse_ratio"] = float("nan")
    df["time_ratio"] = float("nan")
    for model in df["model"].dropna().unique():
        base = df[(df["model"] == model) & (df["run_id"] == "E-Base")]
        if base.empty:
            continue
        base_r2 = float(base.iloc[0]["r2"])
        base_mse = float(base.iloc[0]["mse"])
        base_time = float(base.iloc[0]["train_time_s"])
        idx = df["model"] == model
        df.loc[idx, "r2_retention"] = df.loc[idx, "r2"] / base_r2
        df.loc[idx, "mse_ratio"] = df.loc[idx, "mse"] / base_mse
        df.loc[idx, "time_ratio"] = df.loc[idx, "train_time_s"] / base_time
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect O2 results.")
    parser.add_argument("--results_root", default=str(O2_DIR / "results"))
    parser.add_argument("--o1_eval", default=str(O1_DIR / "results" / "evaluation" / "table4_metrics.csv"))
    parser.add_argument("--o1_runs", default=str(O1_DIR / "results" / "runs"))
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = o1_base_rows(Path(args.o1_eval), Path(args.o1_runs))
    rows.extend(reduced_rows(results_root))
    if not rows:
        print("No O2 rows found.")
        return

    df = pd.DataFrame(rows)
    df = add_relative_metrics(df)
    order = {
        "E-Base": 0,
        "E-AC80": 1,
        "E-AC60": 2,
        "E-AC40": 3,
        "E-AC20": 4,
        "E-AC10": 5,
        "E-AC05": 6,
        "E-GM80": 7,
        "E-GM60": 8,
        "E-GM40": 9,
        "E-GM20": 10,
        "E-J80": 11,
        "E-J60": 12,
        "E-J40": 13,
        "E-J20": 14,
    }
    df["_order"] = df["run_id"].map(order).fillna(999)
    df = df.sort_values(["model", "_order"]).drop(columns=["_order"])

    compiled = results_root / "compiled"
    compiled.mkdir(parents=True, exist_ok=True)
    csv_path = compiled / "o2_data_efficiency_metrics.csv"
    json_path = compiled / "o2_data_efficiency_metrics.json"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
