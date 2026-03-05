# Seismic Response Prediction using Fourier Neural Operators (FNO)

A training framework for FNO-based seismic floor acceleration response prediction.

## Table of Contents
- [Quick Start — Assess Results (No Training)](#quick-start--assess-results-no-training)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models](#models)
- [Scripts](#scripts)
- [Module](#module)
- [Requirements](#requirements)
- [Quick Start — Training](#quick-start--training)

---

## Quick Start — Assess Results (No Training)

To quickly review pre-trained model predictions on the test set **without running any training**, use the included notebook `quick_inference.ipynb` (with results in it).

```
git clone https://github.com/HKUJasonJiang/Seismic-FNO.git
```
### 1. Download the Dataset

Download the KNET seismic dataset (https://huggingface.co/datasets/JasonXF/SeFNO) and place it so that the following paths are accessible:

```
<BASE_DATA_DIR>/MDOF/All_GMs/GMs_knet_3474_AF_57.h5
<BASE_DATA_DIR>/MDOF/knet-250/Data/fno/
```

> **Dataset download:** <!-- TODO: add link -->

By default `BASE_DATA_DIR` in the notebook is set to `D:\BaiduNetdiskDownload\SesimicTransformerData`. Edit Cell 2 of the notebook to point to wherever you place the data.

### 2. Download Pre-trained Models

Download the pre-trained model folders (https://huggingface.co/JasonXF/SeFNO-model) and place them under:

```
output/Paper_Experient/
```

Each model folder should contain a `model/` sub-folder with a `*best*` checkpoint file and a `details/dataset_indices.pkl` file.

> **Model download:** <!-- TODO: add link -->

### 3. Open `quick_inference.ipynb`

Run all cells top-to-bottom. The notebook will:

1. **List** all available models in `output/`
2. **Select** a model by index (edit `MODEL_INDEX` in Cell 4) or by name (`MODEL_NAME`)
3. **Load** the held-out test-set indices that were saved during training
4. **Run inference** on a random (or user-specified) subset of test samples
5. **Plot** time-history and Fourier amplitude spectrum comparisons with R² metrics
6. **Evaluate** on four classical ground motions — El Centro (1940), San Fernando (1971), Michoacán (1985), Kobe (1991) — stored in `test_sample/`

---

## Overview

This project implements a Fourier Neural Operator (FNO v1.0+) for predicting structural floor acceleration response to seismic ground motions (regression task). It includes a standard training script and an efficiency analysis script for studying the effect of dataset size on model performance.

---

## Project Structure

```
SeismicFNO/
├── 1_fno_1_0+_trainscript.py          # Main training script (FNO v1.0+, regression)
├── 1_fno_1_0+_EfficiencyAnalysis.py   # Efficiency analysis training script
├── 0_fno_1_0+_factory.py              # Batch runner for trainscript experiments
├── 0_fno_1_0+_Efficiency_factory.py   # Batch runner for efficiency analysis
├── quick_inference.ipynb              # Notebook: load a model and inspect predictions
├── experiment_summary.csv             # Summary of training experiments
├── efficiency_summary.csv             # Summary of efficiency experiments
├── requirement.txt                    # Python dependencies (Windows)
├── requirement_linux.txt              # Python dependencies (Linux)
├── neuralop/                          # FNO model (From Li et al. github)
├── module/                            # Python scripts
│   ├── dataprep_v2.py                 # Dataset loading (DynamicDataset)
│   ├── train.py                       # Training/validation/test functions
│   └── utility.py                     # Utility functions (device, logging, saving)
└── output/                            # Training outputs (generated at runtime)
    ├── Base-FNO_v1.0+_h64_m64_l4_e50_20251205_063911/       # Baseline model
    ├── Large-FNO_v1.0+_h64_m512_l8_e50_20251206_101922/     # Large model
    ├── Huge-FNO_v1.0+_h128_m1024_l12_e50_20251206_132248/   # Huge model
    ├── Test-Series (Test-1~10)/        # Hyper-parameter sweep (10 runs)
    └── Efficiency-Series (E-Base, E-Test-1~14)/  # Dataset-size study (15 runs)
```

Each model folder contains:
```
<model_folder>/
├── model/
│   ├── fno_best.pth                  # Best checkpoint (lowest val loss)
│   └── fno_epoch_N.pth               # Periodic checkpoints
└── details/
    ├── training_log.csv              # Epoch-by-epoch metrics
    ├── training_config.txt           # Full run configuration
    ├── dataset_indices.pkl           # Train / val / test split indices
    ├── test_results.txt              # Test-set metrics
    └── training_history.png          # Loss / metric curves
```

---

## Models

Three baseline models are provided alongside two experimental series:

### Baseline Models

| Name | Hidden (`h`) | Modes (`m`) | Layers (`l`) | Epochs |
|------|:-----------:|:-----------:|:------------:|:------:|
| **Base** | 64 | 64 | 4 | 50 |
| **Large** | 64 | 512 | 8 | 50 |
| **Huge** | 128 | 1024 | 12 | 50 |

These cover a small, medium, and large FNO capacity range and serve as the primary models evaluated in `quick_inference.ipynb`.

### Experimental Series

| Series | Folder | Runs | Purpose |
|--------|--------|:----:|---------|
| **Test-Series** | `Test-Series (Test-1~10)/` | 10 | Hyper-parameter sweep across modes, layers, and hidden channels |
| **Efficiency-Series** | `Efficiency-Series (E-Base, E-Test-1~15)/` | 16 | Study of training dataset size (number of GMs and scale factors) vs. model performance |

---

## Scripts

### 1. Training Script (`1_fno_1_0+_trainscript.py`)

Main training script for the FNO v1.0+ model (regression only: floor acceleration response prediction). Uses grouped train/val/test splitting to prevent data leakage across scaling factors.

#### Usage:
```bash
python 1_fno_1_0+_trainscript.py [OPTIONS]
```

#### Arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--exp_name` | str | None | Experiment name (used as output folder prefix) |
| `--model_version` | str | "1.0+" | Model version string |
| `--n_modes` | int | 64 | Number of Fourier modes |
| `--hidden_channels` | int | 64 | Number of hidden channels |
| `--n_layers` | int | 4 | Number of FNO layers |
| `--domain_padding` | float | 0.1 | Domain padding fraction |
| `--batch_size` | int | 2560 | Batch size |
| `--epoch` | int | 2 | Number of epochs |
| `--learning_rate` | float | 1e-3 | Learning rate |
| `--weight_decay` | float | 1e-4 | Weight decay |
| `--scheduler_step` | int | 20 | LR scheduler step size |
| `--scheduler_gamma` | float | 0.5 | LR scheduler gamma |
| `--checkpoint_interval` | int | 10 | Save checkpoint every N epochs |
| `--run_test` | bool | True | Run test after training |
| `--disable_tqdm` | flag | False | Disable tqdm progress bars |
| `--aug_factors` | int | 57 | Number of augmentation (scale) factors to use (max 57) |
| `--seed` | int | 42 | Random seed |

#### Example:
```bash
# Basic training with default parameters
python 1_fno_1_0+_trainscript.py

# Custom configuration
python 1_fno_1_0+_trainscript.py --exp_name run1 --n_modes 128 --hidden_channels 64 --epoch 100

# Disable progress bars (e.g. for HPC job logs)
python 1_fno_1_0+_trainscript.py --exp_name hpc_run --epoch 200 --disable_tqdm
```

#### Output:
Saved in `output/[exp_name]-FNO_v1.0+_h{hidden}_m{modes}_l{layers}_e{epochs}_{timestamp}/`
- `model/` — Best model and periodic checkpoints (`.pth`)
- `details/` — Training log (CSV), config file, dataset indices, test results, training curve plot

---

### 2. Efficiency Analysis Script (`1_fno_1_0+_EfficiencyAnalysis.py`)

Trains FNO under controlled dataset conditions to study how the number of ground motions and scale factors affects model performance. Uses nested GM subsets and uniform scale sampling for fair comparison.

#### Usage:
```bash
python 1_fno_1_0+_EfficiencyAnalysis.py [OPTIONS]
```

#### Key Additional Arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_gms_used` | int | 3000 | Number of ground motions to use from the training pool |
| `--use_scales_count` | int | 57 | Number of intensity scales to use per GM |
| `--summary_file` | str | None | CSV file to append results to |

#### Example:
```bash
python 1_fno_1_0+_EfficiencyAnalysis.py --exp_name E-Base --train_gms_used 3000 --use_scales_count 57 --epoch 50
```

---

### 3. Batch Factory Scripts (`0_fno_1_0+_factory.py`, `0_fno_1_0+_Efficiency_factory.py`)

Factory scripts that orchestrate a series of experiments by calling the corresponding `1_*` script with different configurations. Results are automatically logged to a CSV summary file.

#### Usage:
```bash
# Run all training experiments
python 0_fno_1_0+_factory.py

# Run all efficiency experiments (dry-run to preview commands)
python 0_fno_1_0+_Efficiency_factory.py --dry_run

# Run efficiency experiments with custom epochs
python 0_fno_1_0+_Efficiency_factory.py --epochs 100
```

---

## Module

### `dataprep_v2.py` — Dataset Loading

Provides `DynamicDataset`, a PyTorch `Dataset` that lazily loads ground motion and building response data from HDF5 files. Supports index-based splitting for train/val/test partitioning.

```python
from module.dataprep_v2 import DynamicDataset

dataset = DynamicDataset(
    gm_file_path="path/to/GMs.h5",
    building_files_dir="path/to/buildings/",
    gm_indices=train_indices   # optional numpy array
)
```

Each sample returns: `(gm_data, blg_attributes, acc_floor_response, blg_damage_state)`

---

### `train.py` — Training Functions

Provides regression training, validation and test loop functions, plus a training history plot utility.

| Function | Description |
|----------|-------------|
| `train_epoch_regression` | One training epoch (MSE loss, AdamW) |
| `validate_regression` | Validation pass with MSE, RMSE, MAE, R² metrics |
| `test_model_regression` | Test pass with full regression metrics |
| `plot_training_history_regression` | Plots MSE, RMSE, MAE, R² curves from log CSV |

---

### `utility.py` — Utility Functions

| Function / Class | Description |
|-----------------|-------------|
| `Colors` | ANSI color constants for console output |
| `print_colored` | Print colored text (or return as string) |
| `construct_platform_dir` | Resolve base data directory for Windows/Linux |
| `create_output_folder` | Create timestamped output directory |
| `create_log_file` | Initialize training log CSV |
| `save_model` | Save model checkpoint (best or periodic) |
| `save_dataset_indices` | Save train/val/test split indices for reproducibility |
| `count_h5_files` | Count `.h5` files in a directory |
| `set_device` | Detect and report CUDA/CPU device |

---

## Requirements

### Windows
```
pip install -r requirement.txt
```

### Linux
```
pip install -r requirement_linux.txt
```

### Key Dependencies
- `torch` ≥ 2.0  
- `neuralop` (for FNO model)  
- `h5py`, `numpy`, `scikit-learn`, `tqdm`, `colorama`, `matplotlib`

---

## Quick Start — Training

```bash
# 1. Train a model
python 1_fno_1_0+_trainscript.py --exp_name baseline --n_modes 64 --epoch 100

# 2. Run efficiency analysis experiments (via factory)
python 0_fno_1_0+_Efficiency_factory.py --epochs 50

# 3. Check results in the output/ folder and experiment_summary.csv / efficiency_summary.csv
```

---


