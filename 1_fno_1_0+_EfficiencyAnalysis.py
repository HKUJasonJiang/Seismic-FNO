"""
Author: Jason Jiang

Date: 2025/11/10

FNO v1.0 Efficiency Analysis Training Script.
Based on 1_fno_1_0+_trainscript.py.

Date: 08/12/2025 by Copilot

Modified to implement specific data splitting logic for efficiency analysis:
- Nested subsets for Ground Motions (GMs)
- Uniform sampling for Intensity Scales
- Consistent Test Set across experiments
"""

import os
import sys
import time
import glob
import torch
import random
import platform
import argparse
import warnings
import csv
import numpy as np
from colorama import init
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

# Import neuralop FNO model
from neuralop.models import FNO

# MODIFIED: Import from dataprep_v2
from module.dataprep_v2 import DynamicDataset
from module.train import train_epoch_regression, validate_regression, test_model_regression, plot_training_history_regression
from module.utility import (
    Colors, print_colored, construct_platform_dir, create_output_folder,
    create_log_file, save_model, count_h5_files, set_device,
    save_dataset_indices
)

# ignoring warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
init(autoreset=True)

def get_experiment_indices(total_gms=3474, total_scales=57, test_size=474, train_gms_used=3000, use_scales_count=57, seed=42):
    """
    Generates train, validation, and test indices for seismic dataset efficiency experiments.
    
    Args:
        total_gms (int): Total number of Ground Motions available (default: 3474).
        total_scales (int): Total number of scales per GM (default: 57).
        test_size (int): Number of GMs reserved for the test set (default: 474).
        train_gms_used (int): Number of GMs from the pool to use for training/validation (default: 3000).
        use_scales_count (int): Number of scales to use per GM (default: 57).
        seed (int): Random seed for reproducibility (default: 42).
        
    Returns:
        tuple: (train_indices, val_indices, test_indices, train_gm_subset, val_gm_subset, test_gm_indices)
    """
    # Step 1: Global Shuffling (Consistency is key)
    rng = np.random.RandomState(seed)
    shuffled_gms = rng.permutation(total_gms)
    
    # Test Set: Take the last test_size indices. These must never change regardless of other parameters.
    test_gm_indices = shuffled_gms[-test_size:]
    
    # Train/Val Pool: Take the remaining indices.
    pool_gm_indices = shuffled_gms[:-test_size]
    
    # Step 2: Sub-sampling GMs (Nested Subsets)
    # From pool_indices, strictly slice the first train_gms_used indices.
    if train_gms_used > len(pool_gm_indices):
        raise ValueError(f"train_gms_used ({train_gms_used}) cannot be larger than available pool ({len(pool_gm_indices)})")
    
    selected_pool_gms = pool_gm_indices[:train_gms_used]
    
    # Split Train/Val: Perform a standard 80/20 random split on these selected GMs.
    # We use sklearn's train_test_split with the same seed for consistency within this subset
    train_gm_subset, val_gm_subset = train_test_split(
        selected_pool_gms, 
        test_size=0.2, 
        random_state=seed
    )
    
    # Step 3: Sub-sampling Scales (Uniform Sampling)
    if use_scales_count >= total_scales:
        scale_indices = np.arange(total_scales)
    else:
        # use np.linspace ensuring 0 and 56 (last index) are included
        scale_indices = np.linspace(0, total_scales - 1, use_scales_count)
        scale_indices = np.round(scale_indices).astype(int)
        scale_indices = np.unique(scale_indices) # Ensure uniqueness
        
    # Step 4: Flattening
    # Convert the selected GM indices and Scale indices back to the original flattened vector indices
    def flatten_indices(gm_indices, scale_indices, total_scales_constant):
        # Vectorized approach for efficiency
        # shape: (num_gms, num_scales) -> flattened
        # indices = gm * 57 + scale
        gm_grid, scale_grid = np.meshgrid(gm_indices, scale_indices, indexing='ij')
        return (gm_grid * total_scales_constant + scale_grid).flatten()

    train_indices = flatten_indices(train_gm_subset, scale_indices, total_scales)
    val_indices = flatten_indices(val_gm_subset, scale_indices, total_scales)
    
    # MODIFIED: Test set always uses ALL scales (0-56) to ensure fair comparison across experiments
    # regardless of how many scales are used for training.
    all_scales = np.arange(total_scales)
    test_indices = flatten_indices(test_gm_indices, all_scales, total_scales)
    
    return train_indices, val_indices, test_indices, train_gm_subset, val_gm_subset, test_gm_indices


# ==================== Main Script ====================
if __name__ == "__main__":
    # ==================== Arguments ====================
    parser = argparse.ArgumentParser(description='FNO v1.0 Efficiency Analysis Training Script.')
    
    parser.add_argument('--exp_name', type=str, default="EfficiencyAnalysis", help='experiment name (used as folder prefix)')
    parser.add_argument('--model_version', type=str, default="1.0+", help='version number')
    parser.add_argument('--n_modes', type=int, default=1536, help='number of Fourier modes')
    parser.add_argument('--hidden_channels', type=int, default=64, help='hidden channels')
    parser.add_argument('--n_layers', type=int, default=4, help='number of FNO layers')
    parser.add_argument('--domain_padding', type=float, default=0.1, help='domain padding fraction')
    parser.add_argument('--batch_size', type=int, default=2560, help='batch size')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--scheduler_step', type=int, default=20, help='LR scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='LR scheduler gamma')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='save checkpoint every N epochs')
    parser.add_argument('--run_test', type=bool, default=True, help='run test on peer-65 dataset after training')
    parser.add_argument('--disable_tqdm', action='store_true', help='disable tqdm progress bars')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # New arguments for Efficiency Analysis
    parser.add_argument('--train_gms_used', type=int, default=3000, help='Number of GMs from the pool to use (max 3000)')
    parser.add_argument('--use_scales_count', type=int, default=57, help='Number of scales to use per GM (max 57)')
    parser.add_argument('--test_size', type=int, default=474, help='Number of GMs reserved for test set')
    parser.add_argument('--summary_file', type=str, default=None, help='Path to summary CSV file to append results')
    
    args = parser.parse_args()
    
    # ==================== Hyperparameters ====================
    EXP_NAME = args.exp_name
    MODEL_VERSION = args.model_version
    N_MODES = args.n_modes
    HIDDEN_CHANNELS = args.hidden_channels
    N_LAYERS = args.n_layers
    DOMAIN_PADDING = args.domain_padding
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epoch
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    SCHEDULER_STEP = args.scheduler_step
    SCHEDULER_GAMMA = args.scheduler_gamma
    CHECKPOINT_INTERVAL = args.checkpoint_interval
    RUN_TEST = args.run_test
    DISABLE_TQDM = args.disable_tqdm
    SEED = args.seed
    
    # Efficiency Analysis Params
    TRAIN_GMS_USED = args.train_gms_used
    USE_SCALES_COUNT = args.use_scales_count
    TEST_SIZE = args.test_size
    TOTAL_GMS = 3474
    TOTAL_SCALES = 57

    # ==================== Set Random Seed ====================
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Constants
    TIME_POINTS = 3000
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    PROJECTION_RATIO = 2
    
    # ==================== Paths ====================
    PLATFORM = platform.system().lower()
    base_dir = construct_platform_dir(PLATFORM)
    print(f"Base directory: {base_dir}")
    
    GM_TRAIN_PATH = os.path.join(base_dir, 'MDOF', 'All_GMs', 'GMs_knet_3474_AF_57.h5')
    
    BLG_TRAIN_DIR = os.path.join(base_dir, 'MDOF', 'knet-250', 'Data', 'fno')
    
    cwd = os.getcwd()
    SAVE_ROOT_DIR = os.path.join(cwd, 'output')
    
    num_h5_train = count_h5_files(BLG_TRAIN_DIR)
    print(f"Found {num_h5_train} .h5 files in training directory: {BLG_TRAIN_DIR}")
    
    # ==================== Device Setup ====================
    device = set_device()
    
    # ==================== Print Configuration ====================
    print(f"\n{'='*70}")
    print_colored("FNO v1.0 Efficiency Analysis Configuration", Colors.CYAN)
    print(f"{'='*70}")
    print(f"\nModel Architecture:")
    print(f"  - Type: Standard FNO (neuralop)")
    print(f"  - Version: {MODEL_VERSION}")
    print(f"  - Fourier Modes: {N_MODES}")
    print(f"  - Hidden Channels: {HIDDEN_CHANNELS}")
    print(f"  - Number of Layers: {N_LAYERS}")
    print(f"  - Domain Padding: {DOMAIN_PADDING}")
    print(f"  - Projection Ratio: {PROJECTION_RATIO}")
    print(f"  - Grid Size: {TIME_POINTS}")
    
    print(f"\nTraining Parameters:")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Weight Decay: {WEIGHT_DECAY}")
    print(f"  - LR Scheduler: StepLR(step={SCHEDULER_STEP}, gamma={SCHEDULER_GAMMA})")
    
    print(f"\nEfficiency Analysis Parameters:")
    print(f"  - Total GMs: {TOTAL_GMS}")
    print(f"  - Test Size (Fixed): {TEST_SIZE}")
    print(f"  - Train GMs Used (from pool): {TRAIN_GMS_USED}")
    print(f"  - Scales Used per GM: {USE_SCALES_COUNT} / {TOTAL_SCALES}")
    
    print(f"\nDataset Paths:")
    print(f"  - Train/Val/Test GM: {GM_TRAIN_PATH}")
    print(f"  - Train/Val/Test Buildings: {BLG_TRAIN_DIR}")
    
    print(f"\nOutput:")
    print(f"  - Save Directory: {SAVE_ROOT_DIR}")
    print(f"{'='*70}\n")
    
    # ==================== Dataset Preparation ====================
    print_colored("\nPreparing datasets using Efficiency Analysis Logic...", Colors.GREEN)
    
    train_indices, val_indices, test_indices, train_gm_subset, val_gm_subset, test_gm_indices = get_experiment_indices(
        total_gms=TOTAL_GMS,
        total_scales=TOTAL_SCALES,
        test_size=TEST_SIZE,
        train_gms_used=TRAIN_GMS_USED,
        use_scales_count=USE_SCALES_COUNT,
        seed=SEED
    )
    
    # ==================== Verification ====================
    print_colored("\nVerifying split integrity...", Colors.CYAN)
    # Check for intersections
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    assert len(train_set.intersection(val_set)) == 0, "Error: Train and Val sets overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Error: Train and Test sets overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Error: Val and Test sets overlap!"
    
    # Check that original GMs are distinct
    train_orig_set = set(train_gm_subset)
    val_orig_set = set(val_gm_subset)
    test_orig_set = set(test_gm_indices)
    
    assert len(train_orig_set.intersection(val_orig_set)) == 0, "Error: Train and Val original GMs overlap!"
    assert len(train_orig_set.intersection(test_orig_set)) == 0, "Error: Train and Test original GMs overlap!"
    assert len(val_orig_set.intersection(test_orig_set)) == 0, "Error: Val and Test original GMs overlap!"
    
    print("  ✓ No overlap between Train, Val, and Test indices.")
    print("  ✓ No overlap between Original Ground Motions in each set.")
    print("  ✓ Split integrity confirmed.")

    print(f"Dataset Statistics:")
    print(f"  - Train samples: {len(train_indices)} (GMs: {len(train_gm_subset)})")
    print(f"  - Val samples: {len(val_indices)} (GMs: {len(val_gm_subset)})")
    print(f"  - Test samples: {len(test_indices)} (GMs: {len(test_gm_indices)})")
    
    # Create training dataset (knet)
    train_dataset = DynamicDataset(
        gm_file_path=GM_TRAIN_PATH,
        building_files_dir=BLG_TRAIN_DIR,
        gm_indices=train_indices
    )
    
    # Create validation dataset (knet)
    val_dataset = DynamicDataset(
        gm_file_path=GM_TRAIN_PATH,
        building_files_dir=BLG_TRAIN_DIR,
        gm_indices=val_indices
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Must be 0 because h5py objects cannot be pickled for multiprocessing
        pin_memory=True if device == "cuda" else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Must be 0 because h5py objects cannot be pickled for multiprocessing
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(val_dataloader)}")
    
    # ==================== Model Definition ====================
    print_colored("\nDefining Standard FNO model for regression...", Colors.GREEN)
    
    # Create standard FNO for regression only
    model = FNO(
        n_modes=(N_MODES,),
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        projection_channel_ratio=PROJECTION_RATIO,
        n_layers=N_LAYERS,
        domain_padding=DOMAIN_PADDING,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: Standard FNO for Regression")
    print(f"  Task: Floor Acceleration Response Prediction")
    print(f"  Input: Ground Motion Time Series ({TIME_POINTS} time points)")
    print(f"  Output: Floor Acceleration Response ({TIME_POINTS} time points)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compile model for Linux
    if PLATFORM == "linux":
        print_colored("Compiling model for Linux...", Colors.GREEN)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = torch.compile(model)
    
    # ==================== Training Setup ====================
    print_colored("\nSetting up training components...", Colors.GREEN)
    
    criterion = MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = StepLR(
        optimizer,
        step_size=SCHEDULER_STEP,
        gamma=SCHEDULER_GAMMA
    )
    
    print(f"Loss function: MSE Loss")
    print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"Scheduler: StepLR (step={SCHEDULER_STEP}, gamma={SCHEDULER_GAMMA})")
    
    # ==================== Create Output Folder ====================
    # Append efficiency params to exp_name for clarity
    exp_name_full = EXP_NAME
    
    save_dir = create_output_folder(
        version=MODEL_VERSION,
        hidden_channels=HIDDEN_CHANNELS,
        n_modes=N_MODES,
        n_layers=N_LAYERS,
        num_epochs=NUM_EPOCHS,
        save_root_dir=SAVE_ROOT_DIR,
        exp_name=exp_name_full,
    )
    
    # Create subdirectories
    details_dir = os.path.join(save_dir, 'details')
    model_dir = os.path.join(save_dir, 'model')
    os.makedirs(details_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    log_filename = create_log_file(details_dir, task_type='regression')
    print(f"Outputs will be saved to: {save_dir}")
    print(f"  - Details (logs, configs): {details_dir}")
    print(f"  - Models (checkpoints): {model_dir}")
    print(f"Training log: {log_filename}")
    
    # ==================== Save Dataset Indices ====================
    save_dataset_indices(
        save_dir=details_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        total_gm_count=TOTAL_GMS * TOTAL_SCALES,
        split_ratio=[0, 0, 0], # Placeholder as split is custom
        random_seed=SEED
    )
    
    # ==================== Save Training Configuration ====================
    config_file_path = os.path.join(details_dir, 'training_config.txt')
    with open(config_file_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FNO v1.0 Efficiency Analysis Configuration\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("EFFICIENCY ANALYSIS PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total GMs:            {TOTAL_GMS}\n")
        f.write(f"Test Size (Fixed):    {TEST_SIZE}\n")
        f.write(f"Train GMs Used:       {TRAIN_GMS_USED}\n")
        f.write(f"Scales Used per GM:   {USE_SCALES_COUNT}\n")
        f.write(f"Random Seed:          {SEED}\n\n")
        
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Model Version:        {MODEL_VERSION}\n")
        f.write(f"Model Type:           Standard FNO (neuralop)\n")
        f.write(f"Fourier Modes:        {N_MODES}\n")
        f.write(f"Hidden Channels:      {HIDDEN_CHANNELS}\n")
        f.write(f"Number of Layers:     {N_LAYERS}\n")
        f.write(f"Domain Padding:       {DOMAIN_PADDING}\n")
        f.write(f"Projection Ratio:     {PROJECTION_RATIO}\n")
        f.write(f"Input Channels:       {IN_CHANNELS}\n")
        f.write(f"Output Channels:      {OUT_CHANNELS}\n")
        f.write(f"Grid Size:            {TIME_POINTS}\n")
        f.write(f"Total Parameters:     {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
        
        f.write("TRAINING PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Batch Size:           {BATCH_SIZE}\n")
        f.write(f"Number of Epochs:     {NUM_EPOCHS}\n")
        f.write(f"Learning Rate:        {LEARNING_RATE}\n")
        f.write(f"Weight Decay:         {WEIGHT_DECAY}\n")
        f.write(f"Optimizer:            AdamW\n")
        f.write(f"Loss Function:        MSE Loss\n")
        f.write(f"LR Scheduler:         StepLR\n")
        f.write(f"  - Step Size:        {SCHEDULER_STEP}\n")
        f.write(f"  - Gamma:            {SCHEDULER_GAMMA}\n")
        f.write(f"Checkpoint Interval:  {CHECKPOINT_INTERVAL} epochs\n\n")
        
        f.write("DATASET CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train Samples:        {len(train_indices)}\n")
        f.write(f"Validation Samples:   {len(val_indices)}\n")
        f.write(f"Test Samples:         {len(test_indices)}\n")
        f.write(f"Total GM Count:       {TOTAL_GMS * TOTAL_SCALES}\n\n")
        
        f.write("DATA PATHS\n")
        f.write("-" * 70 + "\n")
        f.write(f"GM Path:              {GM_TRAIN_PATH}\n")
        f.write(f"Buildings Dir:        {BLG_TRAIN_DIR}\n")
        f.write(f"Buildings Files:      {num_h5_train} .h5 files\n")
        f.write(f"Output Directory:     {save_dir}\n\n")
        
        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Platform:             {PLATFORM}\n")
        f.write(f"Device:               {device}\n")
        f.write(f"PyTorch Version:      {torch.__version__}\n")
        if device == "cuda":
            f.write(f"CUDA Available:       True\n")
            f.write(f"CUDA Version:         {torch.version.cuda}\n")
            f.write(f"GPU Count:            {torch.cuda.device_count()}\n")
            if torch.cuda.device_count() > 0:
                f.write(f"GPU Device:           {torch.cuda.get_device_name(0)}\n")
        else:
            f.write(f"CUDA Available:       False\n")
        f.write(f"\n")
        
        f.write("ADDITIONAL NOTES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Run Test:             {RUN_TEST}\n")
        f.write(f"Task Type:            Regression (Floor Acceleration Response)\n")
        f.write(f"Compiled Model:       {True if PLATFORM == 'linux' else False}\n")
        f.write(f"DataLoader Workers:   0 (h5py compatibility)\n")
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"Training configuration saved to: {config_file_path}")
    
    # ==================== Training Loop ====================
    print_colored("\nStarting training...", Colors.GREEN)
    print(f"{'='*70}")
    
    best_val_loss = float('inf')
    train_start_time = time.time()
    
    # Create progress bar or simple range iterator based on DISABLE_TQDM flag
    if DISABLE_TQDM:
        epoch_iterator = range(NUM_EPOCHS)
    else:
        epoch_iterator = tqdm(range(NUM_EPOCHS), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_iterator:
        epoch_start_time = time.time()
        
        # Training
        train_metrics = train_epoch_regression(
            model, train_dataloader, criterion, optimizer, device
        )
        
        # Validation
        val_metrics = validate_regression(
            model, val_dataloader, criterion, device
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Update progress bar with metrics or print to console
        if DISABLE_TQDM:
            # Print epoch info to console
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                  f"Train MSE: {train_metrics['mse']:.6f} | "
                  f"Train R²: {train_metrics['r2']:.4f} | "
                  f"Val MSE: {val_metrics['mse']:.6f} | "
                  f"Val R²: {val_metrics['r2']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
        else:
            # Update progress bar
            epoch_iterator.set_postfix({
                'Train_MSE': f'{train_metrics["mse"]:.6f}',
                'Train_R²': f'{train_metrics["r2"]:.4f}',
                'Val_MSE': f'{val_metrics["mse"]:.6f}',
                'Val_R²': f'{val_metrics["r2"]:.4f}',
                'LR': f'{current_lr:.2e}',
                'Time': f'{epoch_time:.1f}s'
            })
        
        # Log to file
        with open(log_filename, 'a') as f:
            f.write(f"{epoch+1},"
                   f"{train_metrics['mse']:.6f},{train_metrics['rmse']:.6f},"
                   f"{train_metrics['mae']:.6f},{train_metrics['r2']:.6f},"
                   f"{val_metrics['mse']:.6f},{val_metrics['rmse']:.6f},"
                   f"{val_metrics['mae']:.6f},{val_metrics['r2']:.6f},"
                   f"{current_lr:.2e},{epoch_time:.2f}\n")
        
        # Save best model
        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            
            # Construct version string with experiment name for model filename
            model_version_for_save = f"{exp_name_full}_{MODEL_VERSION}" if exp_name_full else MODEL_VERSION
            
            model_path = save_model(
                model, optimizer, epoch+1, val_metrics,
                save_dir, model_version_for_save, is_best=True, task_type='regression'
            )
            msg = print_colored(f"  → Best model saved! (Val MSE: {best_val_loss:.6f}, "
                         f"Val RMSE: {val_metrics['rmse']:.6f}, Val R²: {val_metrics['r2']:.4f})", 
                         Colors.GREEN, return_str=True)
            if DISABLE_TQDM:
                print(msg)
            else:
                tqdm.write(msg)
        
        # Save checkpoint every N epochs
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            # Construct version string with experiment name for model filename
            model_version_for_save = f"{exp_name_full}_{MODEL_VERSION}" if exp_name_full else MODEL_VERSION
            
            model_path = save_model(
                model, optimizer, epoch+1, val_metrics,
                model_dir, model_version_for_save, is_best=False, task_type='regression'
            )
            msg = print_colored(f"  → Checkpoint saved at epoch {epoch+1}", Colors.CYAN, return_str=True)
            if DISABLE_TQDM:
                print(msg)
            else:
                tqdm.write(msg)
    
    # ==================== Training Complete ====================
    total_train_time = time.time() - train_start_time
    
    print(f"{'='*70}")
    print_colored("Training complete!", Colors.GREEN)
    print(f"Total training time: {total_train_time:.2f}s ({total_train_time/60:.2f} min)")
    print(f"Best validation MSE: {best_val_loss:.6f}")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*70}")
    
    # ==================== Plot Training History ====================
    print_colored("\nGenerating training history plots...", Colors.GREEN)
    try:
        plot_path = plot_training_history_regression(log_filename, details_dir)
        print(f"Training history plot saved to: {plot_path}")
    except Exception as e:
        print_colored(f"Warning: Failed to generate training history plot: {e}", Colors.YELLOW)
    
    # ==================== Test Model ====================
    if RUN_TEST:
        # Load best model for testing
        print_colored("\nLoading best model for testing...", Colors.GREEN)
        
        # Construct version string to match how it was saved
        model_version_for_save = f"{exp_name_full}_{MODEL_VERSION}" if exp_name_full else MODEL_VERSION
        version_display = model_version_for_save.replace('_', '.')
        
        best_model_pattern = os.path.join(save_dir, f"fno_v{version_display}_best_*.pth")
        best_model_files = glob.glob(best_model_pattern)
        
        if best_model_files:
            # Should only be one best model now (no timestamp)
            best_model_path = best_model_files[0]
            print(f"Loading best model: {os.path.basename(best_model_path)}")
        else:
            # Fallback search if exact match fails (sometimes version string formatting varies)
            print(f"Warning: Exact match not found for {best_model_pattern}. Searching for any best model in dir.")
            best_model_files = glob.glob(os.path.join(save_dir, "*best*.pth"))
            if best_model_files:
                best_model_path = best_model_files[0]
                print(f"Loading best model: {os.path.basename(best_model_path)}")
            else:
                raise FileNotFoundError(f"No best model found in: {save_dir}")
        
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        print(f"\n{'='*70}")
        print_colored("Starting Model Evaluation on Test Sets", Colors.GREEN)
        print(f"{'='*70}\n")

        # ========== Prepare the test dataset ==========
        print_colored("\nPreparing test datasets...", Colors.GREEN)
        
        # Test dataset from knet (Fixed Test Set)
        test_dataset = DynamicDataset(
            gm_file_path=GM_TRAIN_PATH,
            building_files_dir=BLG_TRAIN_DIR,
            gm_indices=test_indices
        )
        
        print(f"\nTest dataset composition:")
        print(f"  - Test split size: {len(test_dataset)}")
        
        # Create dataloader for test
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # Must be 0 because h5py objects cannot be pickled for multiprocessing
            pin_memory=True if device == "cuda" else False
        )
        
        print(f"Number of test batches: {len(test_dataloader)}")
        
        # ========== Test Mode: Test Split ==========
        print_colored("\nTEST MODE: Fixed Test Split", Colors.CYAN)
        print(f"{'-'*70}")
        
        test_metrics = test_model_regression(
            model, test_dataloader, criterion, device
        )
        
        print(f"\n{'='*70}")
        print_colored("Test Results (Fixed Test Split):", Colors.BLUE)
        print(f"\n  REGRESSION TASK (Floor Acceleration Response):")
        print(f"    - MSE Loss:  {test_metrics['mse']:.6f}")
        print(f"    - RMSE:      {test_metrics['rmse']:.6f}")
        print(f"    - MAE:       {test_metrics['mae']:.6f}")
        print(f"    - R² Score:  {test_metrics['r2']:.6f}")
        print(f"{'='*70}\n")
        
        # ========== Save Test Results ==========
        print_colored("Saving test results...", Colors.GREEN)
        
        # Save combined results to results.txt
        results_path = os.path.join(details_dir, 'results.txt')
        avg_epoch_time = total_train_time / NUM_EPOCHS
        
        with open(results_path, 'w') as f:
            f.write("Regression Test Results - Efficiency Analysis\n")
            f.write(f"{'='*60}\n")
            f.write(f"Model Version: {MODEL_VERSION} (Standard FNO)\n\n")
            
            f.write(f"Efficiency Analysis:\n")
            f.write(f"  - Train GMs Used: {TRAIN_GMS_USED}\n")
            f.write(f"  - Scales Used: {USE_SCALES_COUNT}\n\n")
            
            f.write(f"Model Configuration:\n")
            f.write(f"  - Hidden Channels: {HIDDEN_CHANNELS}\n")
            f.write(f"  - Fourier Modes: {N_MODES}\n")
            f.write(f"  - Number of Layers: {N_LAYERS}\n")
            f.write(f"  - Domain Padding: {DOMAIN_PADDING}\n\n")
            
            f.write(f"REGRESSION (Floor Acceleration Response)\n")
            f.write(f"{'-'*60}\n")
            f.write(f"MSE Loss:  {test_metrics['mse']:.6f}\n")
            f.write(f"RMSE:      {test_metrics['rmse']:.6f}\n")
            f.write(f"MAE:       {test_metrics['mae']:.6f}\n")
            f.write(f"R² Score:  {test_metrics['r2']:.6f}\n")
            f.write(f"\n{'='*60}\n")
            
            f.write(f"Training Performance:\n")
            f.write(f"  - Total Training Time: {total_train_time:.2f}s\n")
            f.write(f"  - Average Epoch Time:  {avg_epoch_time:.2f}s\n")
            f.write(f"{'='*60}\n")
        
        print(f"  Results saved to: {results_path}")
        
        # ========== Append to Summary CSV ==========
        if args.summary_file:
            file_exists = os.path.isfile(args.summary_file)
            with open(args.summary_file, 'a', newline='') as csvfile:
                fieldnames = ['ExpName', 'ModelVersion', 'TrainGMs', 'Scales', 'Epochs', 'BatchSize', 'MSE', 'RMSE', 'MAE', 'R2', 'TrainingTime']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'ExpName': EXP_NAME,
                    'ModelVersion': MODEL_VERSION,
                    'TrainGMs': TRAIN_GMS_USED,
                    'Scales': USE_SCALES_COUNT,
                    'Epochs': NUM_EPOCHS,
                    'BatchSize': BATCH_SIZE,
                    'MSE': f"{test_metrics['mse']:.6f}",
                    'RMSE': f"{test_metrics['rmse']:.6f}",
                    'MAE': f"{test_metrics['mae']:.6f}",
                    'R2': f"{test_metrics['r2']:.6f}",
                    'TrainingTime': f"{total_train_time:.2f}"
                })
            print_colored(f"Summary appended to: {args.summary_file}", Colors.GREEN)

        print(f"\n{'='*70}")
        print_colored("All test results saved successfully!", Colors.GREEN)
        print(f"{'='*70}")
