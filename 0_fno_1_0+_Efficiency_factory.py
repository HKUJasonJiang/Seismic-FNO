"""
Author: Jason Jiang
Date: 2025/12/08

Factory script to manage FNO v1.0 Efficiency Analysis experiments.
Runs 1_fno_1_0+_EfficiencyAnalysis.py with different configurations based on Table 3.
"""

import os
import subprocess
import time
import argparse
import sys

def run_experiment(config, summary_file, dry_run=False):
    # Construct command
    # Assuming the script is in the same directory
    script_path = "1_fno_1_0+_EfficiencyAnalysis.py"
    
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found!")
        return

    cmd = [
        sys.executable, script_path,
        "--exp_name", config["name"],
        "--train_gms_used", str(config["gms"]),
        "--use_scales_count", str(config["scales"]),
        "--epoch", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--model_version", "1.0+_E",
        "--summary_file", summary_file
    ]
    
    print(f"\n{'='*80}")
    print(f"Running Experiment: {config['name']}")
    print(f"Configuration: GMs={config['gms']}, Scales={config['scales']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    if not dry_run:
        start_time = time.time()
        try:
            subprocess.run(cmd, check=True)
            duration = time.time() - start_time
            print(f"\nExperiment {config['name']} completed in {duration/60:.2f} minutes.")
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment {config['name']}: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factory script for Efficiency Analysis experiments")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for each experiment")
    parser.add_argument("--batch_size", type=int, default=2560, help="Batch size")
    parser.add_argument("--filter", type=str, default=None, help="Filter experiments by name (e.g., 'E-test-1')")
    
    args = parser.parse_args()

    # Define experiments based on Table 3
    # E-Base: 3000 GMs, 57 Scales
    # E-test-1 to 4: Varying Scales (80%, 60%, 40%, 20%)
    # E-test-5 to 8: Varying GMs (80%, 60%, 40%, 20%)
    experiments = [
        {"name": "E-Base",   "gms": 3000, "scales": 57},
        {"name": "E-test-1", "gms": 3000, "scales": 46}, # 80% scales
        {"name": "E-test-2", "gms": 3000, "scales": 34}, # 60% scales
        {"name": "E-test-3", "gms": 3000, "scales": 23}, # 40% scales
        {"name": "E-test-4", "gms": 3000, "scales": 11}, # 20% scales
        {"name": "E-test-5", "gms": 2400, "scales": 57}, # 80% GMs
        {"name": "E-test-6", "gms": 1800, "scales": 57}, # 60% GMs
        {"name": "E-test-7", "gms": 1200, "scales": 57}, # 40% GMs
        {"name": "E-test-8", "gms": 600,  "scales": 57}, # 20% GMs
        {"name": "E-test-9", "gms": 2400,  "scales": 46}, # 80%
        {"name": "E-test-10", "gms": 1800, "scales": 34}, # 60%
        {"name": "E-test-11", "gms": 1200, "scales": 23}, # 40%
        {"name": "E-test-12", "gms": 600,  "scales": 11}, # 20%
        {"name": "E-test-13", "gms": 3000, "scales": 6}, # 20% scales
        {"name": "E-test-14", "gms": 3000, "scales": 3}, # 20% scales
        {"name": "E-test-15", "gms": 3000, "scales": 1}, # 20% scales
    ]

    # Add common config
    for exp in experiments:
        exp["epochs"] = args.epochs
        exp["batch_size"] = args.batch_size

    print(f"Found {len(experiments)} defined experiments.")
    
    # Filter experiments if requested
    selected_experiments = []
    for exp in experiments:
        if args.filter:
            if args.filter in exp["name"]:
                selected_experiments.append(exp)
        else:
            selected_experiments.append(exp)
            
    print(f"Selected {len(selected_experiments)} experiments to run.")

    # Run experiments
    summary_file = os.path.abspath("efficiency_summary.csv")
    for i, exp in enumerate(selected_experiments):
        print(f"\nProcessing {i+1}/{len(selected_experiments)}...")
        try:
            run_experiment(exp, summary_file, dry_run=args.dry_run)
        except subprocess.CalledProcessError:
            print(f"Experiment {exp['name']} failed. Stopping factory.")
            break
        except KeyboardInterrupt:
            print("\nExecution interrupted by user.")
            break
