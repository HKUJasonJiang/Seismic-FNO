import subprocess
import sys
import os
import argparse
import csv
import time
from datetime import datetime

def get_experiment_results(exp_name):
    """
    Reads the results.txt file from the most recent output folder for the given experiment.
    Returns a dictionary with the results.
    """
    # Find the output directory for this experiment
    # We assume the output directory structure is output/{exp_name}-FNO_v1.0+_...
    # Since we don't know the exact timestamp, we'll look for the most recent folder matching the pattern
    
    output_root = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_root):
        return {}
        
    # List all folders in output directory
    folders = [f for f in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, f))]
    
    # Filter folders that start with the experiment name
    exp_folders = [f for f in folders if f.startswith(f"{exp_name}-FNO")]
    
    if not exp_folders:
        return {}
        
    # Sort by name (which includes timestamp) to get the most recent one
    exp_folders.sort(reverse=True)
    latest_folder = exp_folders[0]
    
    results_file = os.path.join(output_root, latest_folder, 'details', 'results.txt')
    
    results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            content = f.read()
            # Parse the file content
            # Expected format:
            # Regression Test Results - KNET Test Split
            # ...
            # MSE Loss:  0.000123
            # ...
            # R² Score:  0.987654
            # ...
            # Training Performance:
            #   - Total Training Time: 123.45s
            #   - Average Epoch Time:  1.23s
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith("MSE Loss:"):
                    results['mse'] = float(line.split(':')[1].strip())
                elif line.startswith("R² Score:"):
                    results['r2'] = float(line.split(':')[1].strip())
                elif line.startswith("- Total Training Time:"):
                    results['total_time'] = line.split(':')[1].strip()
                elif line.startswith("- Average Epoch Time:"):
                    results['avg_epoch_time'] = line.split(':')[1].strip()
                    
    return results

def log_experiment_to_csv(csv_file, exp_data):
    """
    Logs experiment data to a CSV file.
    """
    file_exists = os.path.exists(csv_file)
    
    fieldnames = [
        'timestamp', 'experiment_name', 'n_modes', 'hidden_channels', 
        'n_layers', 'batch_size', 'epochs', 'seed', 'mse', 'r2', 
        'total_training_time', 'avg_epoch_time'
    ]
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(exp_data)

def run_experiment(exp_name, n_modes, hidden_channels, n_layers, batch_size, epochs, seed=42, csv_file="experiment_log.csv"):
    print(f"==================================================================")
    print(f"Starting Experiment: {exp_name}")
    print(f"Parameters: k_max={n_modes}, H_Channel={hidden_channels}, N_Layer={n_layers}, Batch={batch_size}, Epochs={epochs}, Seed={seed}")
    print(f"==================================================================")
    
    cmd = [
        sys.executable, "1_fno_1_0+_trainscript.py",
        "--exp_name", exp_name,
        "--n_modes", str(n_modes),
        "--hidden_channels", str(hidden_channels),
        "--n_layers", str(n_layers),
        "--batch_size", str(batch_size),
        "--epoch", str(epochs),
        "--seed", str(seed)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Experiment {exp_name} completed successfully.")
        
        # Retrieve results and log to CSV
        results = get_experiment_results(exp_name)
        
        log_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'experiment_name': exp_name,
            'n_modes': n_modes,
            'hidden_channels': hidden_channels,
            'n_layers': n_layers,
            'batch_size': batch_size,
            'epochs': epochs,
            'seed': seed,
            'mse': results.get('mse', 'N/A'),
            'r2': results.get('r2', 'N/A'),
            'total_training_time': results.get('total_time', 'N/A'),
            'avg_epoch_time': results.get('avg_epoch_time', 'N/A')
        }
        
        log_experiment_to_csv(csv_file, log_data)
        print(f"Results logged to {csv_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"Experiment {exp_name} failed with error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Factory script for FNO experiments')
    parser.add_argument('--batch_size', type=int, default=2560, help='Batch size for all experiments')
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs for all experiments')
    parser.add_argument('--csv_log', type=str, default='experiment_summary.csv', help='CSV file to log results')
    args = parser.parse_args()

    # Define experiments based on the provided table
    # No. | k_max | H_Channel | N_Layer | Batch Size
    experiments = [
        {"name": "Base",    "k_max": 64,   "H_Channel": 64,  "N_Layer": 4,  "batch_size": 2560},
        {"name": "Test-1",  "k_max": 128,  "H_Channel": 64,  "N_Layer": 4,  "batch_size": 2560},
        {"name": "Test-2",  "k_max": 512,  "H_Channel": 64,  "N_Layer": 4,  "batch_size": 2560},
        {"name": "Test-3",  "k_max": 1024, "H_Channel": 64,  "N_Layer": 4,  "batch_size": 2560},
        {"name": "Test-4",  "k_max": 1536, "H_Channel": 64,  "N_Layer": 4,  "batch_size": 2560},
        {"name": "Test-5",  "k_max": 64,   "H_Channel": 16,  "N_Layer": 4,  "batch_size": 5120},
        {"name": "Test-6",  "k_max": 64,   "H_Channel": 32,  "N_Layer": 4,  "batch_size": 5120},
        {"name": "Test-7",  "k_max": 64,   "H_Channel": 128, "N_Layer": 4,  "batch_size": 1280},
        {"name": "Test-8",  "k_max": 64,   "H_Channel": 64,  "N_Layer": 2,  "batch_size": 2560},
        {"name": "Test-9",  "k_max": 64,   "H_Channel": 64,  "N_Layer": 8,  "batch_size": 1536},
        {"name": "Test-10", "k_max": 64,   "H_Channel": 64,  "N_Layer": 12, "batch_size": 1152},
        {"name": "Large",   "k_max": 512,  "H_Channel": 64,  "N_Layer": 8,  "batch_size": 1536},
        {"name": "Huge",    "k_max": 1024, "H_Channel": 128, "N_Layer": 12, "batch_size": 512},
    ]
    
    # Set a common seed for all experiments or vary it if needed. 
    # Using 42 as default based on previous script.
    COMMON_SEED = 42
    
    for exp in experiments:
        # Use batch_size from experiment dict if available, otherwise use command line arg
        batch_size = exp.get("batch_size", args.batch_size)
        run_experiment(
            exp_name=exp["name"],
            n_modes=exp["k_max"],
            hidden_channels=exp["H_Channel"],
            n_layers=exp["N_Layer"],
            batch_size=batch_size,
            epochs=args.epoch,
            seed=COMMON_SEED,
            csv_file=args.csv_log
        )
