"""
Author: Jason Jiang
Date: 2025/11/5

Utility functions for FNO training and model management.
This module contains helper functions for device setup, directory management,
model saving/loading, and output formatting.
"""

import os
import torch
from datetime import datetime
from colorama import Fore, Style


class Colors:
    """ANSI color codes for colored console output."""
    BLUE = Fore.BLUE
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    CYAN = Fore.CYAN
    MAGENTA = Fore.MAGENTA


def print_colored(text, color, return_str=False):
    """Print colored text to console.
    
    Args:
        text (str): Text to print
        color: Color code from Colors class (e.g., Colors.GREEN)
        return_str (bool): If True, return the colored string instead of printing
        
    Returns:
        str or None: Colored string if return_str=True, otherwise None
    """
    colored_text = f"{color}{text}{Style.RESET_ALL}"
    if return_str:
        return colored_text
    else:
        print(colored_text)


def construct_platform_dir(platform_name):
    """Construct base directory based on platform.
    
    Args:
        platform_name (str): Platform name ('windows' or 'linux')
        
    Returns:
        base_dir (str): Base directory path for the platform
        
    Raises:
        ValueError: If platform_name is not recognized
    """
    if platform_name == "windows":
        return r'D:\BaiduNetdiskDownload\SesimicTransformerData'
    elif platform_name == "linux":
        return r'/home/jason/SesimicTransformerData'
    else:
        raise ValueError(f"Unknown platform: {platform_name}")


def create_output_folder(version, hidden_channels, n_modes, n_layers, num_epochs, save_root_dir, exp_name=None):
    """Create output folder for saving models and logs.
    
    Args:
        version (str): Model version string (e.g., "1_0" or "1_1")
        hidden_channels (int): Number of hidden channels
        n_modes (int): Number of Fourier modes
        n_layers (int): Number of FNO layers
        num_epochs (int): Number of training epochs
        save_root_dir (str): Root directory for saving outputs
        exp_name (str, optional): Experiment name to prefix the folder name
        
    Returns:
        save_path (str): Path to the created output folder
    """
    # Convert version format from "1_0" to "1.0" for display
    version_display = version.replace('_', '.')
    
    # Generate timestamp for unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build folder name with optional experiment name prefix
    base_name = f"FNO_v{version_display}_h{hidden_channels}_m{n_modes}_l{n_layers}_e{num_epochs}_{timestamp}"
    if exp_name:
        folder_name = f"{exp_name}-{base_name}"
    else:
        folder_name = base_name
    
    save_path = os.path.join(save_root_dir, folder_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    return save_path


def create_log_file(save_dir, task_type='multi-task'):
    """Create log file for training records.
    
    Args:
        save_dir (str): Directory to save the log file
        task_type (str): Type of task - 'multi-task', 'regression', or 'classification'
        
    Returns:
        log_path (str): Path to the created log file
    """
    log_path = os.path.join(save_dir, 'training_log.txt')
    with open(log_path, 'w') as f:
        if task_type == 'multi-task':
            f.write("Epoch,Train_MSE,Train_RMSE,Train_MAE,Train_R2,Train_CE_Loss,Train_Acc,Train_Total_Loss,"
                    "Val_MSE,Val_RMSE,Val_MAE,Val_R2,Val_CE_Loss,Val_Acc,Val_Total_Loss,Learning_Rate,Time(s)\n")
        elif task_type == 'regression':
            f.write("Epoch,Train_MSE,Train_RMSE,Train_MAE,Train_R2,"
                    "Val_MSE,Val_RMSE,Val_MAE,Val_R2,Learning_Rate,Time(s)\n")
        elif task_type == 'classification':
            f.write("Epoch,Train_CE_Loss,Train_Acc,Train_Precision,Train_Recall,Train_F1,"
                    "Val_CE_Loss,Val_Acc,Val_Precision,Val_Recall,Val_F1,Learning_Rate,Time(s)\n")
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Must be 'multi-task', 'regression', or 'classification'.")
    return log_path


def save_model(model, optimizer, epoch, metrics, save_dir, version, is_best=False, task_type='multi-task'):
    """Save model checkpoint with both regression and classification metrics.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        epoch (int): Current epoch number
        metrics (dict): Dictionary of metrics to save
        save_dir (str): Directory to save the model
        version (str): Model version string (e.g., "1_0" or "1_1")
        is_best (bool): Whether this is the best model so far
        task_type (str): Type of task - 'multi-task', 'regression', or 'classification'
        
    Returns:
        model_path (str): Path to the saved model checkpoint
    """
    # Convert version format from "1_0" to "1.0" for display
    version_display = version.replace('_', '.')
    
    if is_best:
        # For best model: no timestamp, include key metrics in filename
        # Remove old best model if it exists
        import glob
        old_best_pattern = os.path.join(save_dir, f"fno_v{version_display}_best_*.pth")
        for old_file in glob.glob(old_best_pattern):
            os.remove(old_file)
        
        # Create filename with key metrics
        if task_type == 'regression':
            # For regression: include MSE and R2
            mse = metrics.get('mse', 0)
            r2 = metrics.get('r2', 0)
            model_path = os.path.join(save_dir, f"fno_v{version_display}_best_mse{mse:.4f}_r2{r2:.4f}.pth")
        elif task_type == 'multi-task':
            # For multi-task: include MSE (regression) and Accuracy (classification)
            mse = metrics.get('reg_loss', 0)  # reg_loss is MSE
            acc = metrics.get('accuracy', 0)
            model_path = os.path.join(save_dir, f"fno_v{version_display}_best_mse{mse:.4f}_acc{acc:.4f}.pth")
        elif task_type == 'classification':
            # For classification: include Loss and Accuracy
            loss = metrics.get('loss', 0)
            acc = metrics.get('accuracy', 0)
            model_path = os.path.join(save_dir, f"fno_v{version_display}_best_loss{loss:.4f}_acc{acc:.4f}.pth")
        else:
            # Fallback for other task types
            model_path = os.path.join(save_dir, f"fno_v{version_display}_best.pth")
    else:
        # For checkpoints: include epoch and timestamp with key metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if task_type == 'regression':
            mse = metrics.get('mse', 0)
            r2 = metrics.get('r2', 0)
            model_path = os.path.join(save_dir, f"fno_v{version_display}_epoch{epoch}_mse{mse:.4f}_r2{r2:.4f}_{timestamp}.pth")
        elif task_type == 'multi-task':
            mse = metrics.get('reg_loss', 0)
            acc = metrics.get('accuracy', 0)
            model_path = os.path.join(save_dir, f"fno_v{version_display}_epoch{epoch}_mse{mse:.4f}_acc{acc:.4f}_{timestamp}.pth")
        elif task_type == 'classification':
            loss = metrics.get('loss', 0)
            acc = metrics.get('accuracy', 0)
            model_path = os.path.join(save_dir, f"fno_v{version_display}_epoch{epoch}_loss{loss:.4f}_acc{acc:.4f}_{timestamp}.pth")
        else:
            model_path = os.path.join(save_dir, f"fno_v{version_display}_epoch{epoch}_{timestamp}.pth")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, model_path)
    
    return model_path


def save_dataset_indices(save_dir, train_indices, val_indices, test_indices, 
                         total_gm_count, split_ratio, random_seed=42):
    """Save dataset split indices for reproducibility.
    
    Args:
        save_dir (str): Directory to save the indices files
        train_indices (numpy.ndarray): Training set indices
        val_indices (numpy.ndarray): Validation set indices
        test_indices (numpy.ndarray): Test set indices
        total_gm_count (int): Total number of ground motions
        split_ratio (list): Split ratio [train, val, test] (e.g., [0.7, 0.2, 0.1])
        random_seed (int): Random seed used for splitting (default: 42)
        
    Returns:
        tuple: Paths to (pickle_file, text_file)
    """
    import pickle
    import numpy as np
    
    # Prepare data dictionary
    indices_data = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'total_gm_count': total_gm_count,
        'split_ratio': split_ratio,
        'random_seed': random_seed
    }
    
    # Save as pickle file
    indices_pkl_path = os.path.join(save_dir, 'dataset_indices.pkl')
    with open(indices_pkl_path, 'wb') as f:
        pickle.dump(indices_data, f)
    
    # Save as text file for easy viewing
    indices_txt_path = os.path.join(save_dir, 'dataset_indices.txt')
    with open(indices_txt_path, 'w') as f:
        f.write("Dataset Split Indices\n")
        f.write("="*60 + "\n\n")
        f.write(f"Random Seed: {random_seed}\n")
        f.write(f"Split Ratio (Train:Val:Test): {split_ratio}\n")
        f.write(f"Total GM Count: {total_gm_count}\n\n")
        
        f.write(f"Train Indices ({len(train_indices)} samples):\n")
        f.write(f"  Range: [{train_indices.min()}, {train_indices.max()}]\n")
        f.write(f"  First 10: {train_indices[:10].tolist()}\n")
        f.write(f"  Last 10: {train_indices[-10:].tolist()}\n\n")
        
        f.write(f"Validation Indices ({len(val_indices)} samples):\n")
        f.write(f"  Range: [{val_indices.min()}, {val_indices.max()}]\n")
        f.write(f"  First 10: {val_indices[:10].tolist()}\n")
        f.write(f"  Last 10: {val_indices[-10:].tolist()}\n\n")
        
        f.write(f"Test Indices ({len(test_indices)} samples):\n")
        f.write(f"  Range: [{test_indices.min()}, {test_indices.max()}]\n")
        f.write(f"  First 10: {test_indices[:10].tolist()}\n")
        f.write(f"  Last 10: {test_indices[-10:].tolist()}\n")
    
    print_colored(f"Dataset indices saved:", Colors.GREEN)
    print(f"  - Pickle file: {indices_pkl_path}")
    print(f"  - Text summary: {indices_txt_path}")
    
    return indices_pkl_path, indices_txt_path


def count_h5_files(directory):
    """Count number of .h5 files in directory.
    
    Args:
        directory (str): Directory path to search for .h5 files
        
    Returns:
        count (int): Number of .h5 files found
    """
    return len([f for f in os.listdir(directory) if f.endswith('.h5')])


def set_device():
    """Set and configure the training device (CUDA or CPU).
    
    Returns:
        device (str): Device string ('cuda' or 'cpu')
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_colored(f"Using device: {device}", Colors.GREEN)
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return device

