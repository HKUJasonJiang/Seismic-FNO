"""
Author: Jason Jiang
Date: 2025/11/5

Training, validation, and testing functions for FNO-based models.
This module contains all training-related functions including epoch training,
validation, testing, and plotting utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

def plot_training_history_regression(log_file, save_dir):
    """Plot training history from log file for regression-only tasks.
    
    Args:
        log_file (str): Path to the training log CSV file
        save_dir (str): Directory to save the plot
        
    Returns:
        plot_path (str): Path to the saved plot
    """
    # Read log file
    df = pd.read_csv(log_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot MSE loss
    axes[0, 0].plot(df['Epoch'], df['Train_MSE'], label='Train', marker='o', markersize=3)
    axes[0, 0].plot(df['Epoch'], df['Val_MSE'], label='Validation', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Mean Squared Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot RMSE
    axes[0, 1].plot(df['Epoch'], df['Train_RMSE'], label='Train', marker='o', markersize=3)
    axes[0, 1].plot(df['Epoch'], df['Val_RMSE'], label='Validation', marker='s', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Root Mean Squared Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot MAE
    axes[1, 0].plot(df['Epoch'], df['Train_MAE'], label='Train', marker='o', markersize=3)
    axes[1, 0].plot(df['Epoch'], df['Val_MAE'], label='Validation', marker='s', markersize=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('Mean Absolute Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot R² Score
    axes[1, 1].plot(df['Epoch'], df['Train_R2'], label='Train', marker='o', markersize=3)
    axes[1, 1].plot(df['Epoch'], df['Val_R2'], label='Validation', marker='s', markersize=3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('R² Score (Coefficient of Determination)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_history_regression.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


# ==================== Regression-Only Functions (v1.0) ====================

def train_epoch_regression(model, train_loader, criterion, optimizer, device):
    """Train for one epoch with regression only (no classification).
    
    Args:
        model: The FNO model for regression
        train_loader: DataLoader for training data
        criterion: Loss criterion for regression task (e.g., MSELoss)
        optimizer: Optimizer for training
        device: Device to run training on ('cuda' or 'cpu')
        
    Returns:
        metrics (dict): Dictionary containing:
            - mse: Average MSE loss
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - r2: R² score
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for gm_data, blg_attributes, acc_floor_response, blg_damage_state in pbar:
        # Move data to device and transpose to (batch, channels, length)
        gm_data = gm_data.permute(0, 2, 1).to(device)
        acc_floor_response = acc_floor_response.permute(0, 2, 1).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(gm_data)
        
        # Compute loss
        loss = criterion(output, acc_floor_response)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Store predictions and targets
        all_preds.append(output.detach().cpu().numpy())
        all_targets.append(acc_floor_response.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / len(train_loader)
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Flatten for metrics calculation
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(targets_flat, preds_flat)
    rmse = np.sqrt(avg_loss)  # RMSE from MSE
    r2 = r2_score(targets_flat, preds_flat)
    
    metrics = {
        'mse': avg_loss,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics


def validate_regression(model, val_loader, criterion, device):
    """Validate the model with regression only (no classification).
    
    Args:
        model: The FNO model for regression
        val_loader: DataLoader for validation data
        criterion: Loss criterion for regression task
        device: Device to run validation on ('cuda' or 'cpu')
        
    Returns:
        metrics (dict): Dictionary containing:
            - mse: Average MSE loss
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - r2: R² score
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for gm_data, blg_attributes, acc_floor_response, blg_damage_state in pbar:
            # Move data to device and transpose to (batch, channels, length)
            gm_data = gm_data.permute(0, 2, 1).to(device)
            acc_floor_response = acc_floor_response.permute(0, 2, 1).to(device)
            
            # Forward pass
            output = model(gm_data)
            
            # Compute loss
            loss = criterion(output, acc_floor_response)
            total_loss += loss.item()
            
            # Store predictions and targets
            all_preds.append(output.cpu().numpy())
            all_targets.append(acc_floor_response.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / len(val_loader)
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Flatten for metrics calculation
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(targets_flat, preds_flat)
    rmse = np.sqrt(avg_loss)  # RMSE from MSE
    r2 = r2_score(targets_flat, preds_flat)
    
    metrics = {
        'mse': avg_loss,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics


def test_model_regression(model, test_loader, criterion, device):
    """Test the model and return detailed regression metrics only.
    
    Args:
        model: The FNO model for regression
        test_loader: DataLoader for test data
        criterion: Loss criterion for regression task
        device: Device to run testing on ('cuda' or 'cpu')
        
    Returns:
        metrics (dict): Dictionary containing:
            - mse: Average MSE loss
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - r2: R² score
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for gm_data, blg_attributes, acc_floor_response, blg_damage_state in pbar:
            # Move data to device and transpose to (batch, channels, length)
            gm_data = gm_data.permute(0, 2, 1).to(device)
            acc_floor_response = acc_floor_response.permute(0, 2, 1).to(device)
            
            # Forward pass
            output = model(gm_data)
            
            # Compute loss
            loss = criterion(output, acc_floor_response)
            total_loss += loss.item()
            
            # Store predictions and targets
            all_preds.append(output.cpu().numpy())
            all_targets.append(acc_floor_response.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / len(test_loader)
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Flatten for metrics calculation
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(targets_flat, preds_flat)
    rmse = np.sqrt(avg_loss)  # RMSE from MSE
    r2 = r2_score(targets_flat, preds_flat)
    
    metrics = {
        'mse': avg_loss,
        'rmse': rmse,
        'mae': mae,
           'r2': r2
    }
    
    return metrics


