"""
Training script for ResNet1D model for HMS Brain Activity Classification.
"""

import os
import sys
import argparse
import yaml
import time
import random
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet1d import EEGNet, EEGDataset
from data.preprocessing import eeg_from_parquet


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int, optional): Random seed. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get PyTorch device to use.
    
    Returns:
        torch.device: Device to use for training
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(train_path, test_path):
    """
    Load training and test data.
    
    Args:
        train_path (str): Path to training CSV file
        test_path (str): Path to test CSV file
        
    Returns:
        tuple: (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Normalize target columns
    target_cols = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
    train_df[target_cols] = train_df[target_cols].div(train_df[target_cols].sum(axis=1), axis=0)
    
    return train_df, test_df


def prepare_eegs(df, eeg_path, eeg_features, config):
    """
    Prepare EEG data from parquet files.
    
    Args:
        df (pd.DataFrame): Dataframe with EEG IDs
        eeg_path (str): Path to EEG directory
        eeg_features (list): List of EEG features to extract
        config (object): Configuration object with sampling parameters
        
    Returns:
        dict: Dictionary mapping EEG IDs to processed EEG data
    """
    all_eegs = {}
    
    # Get unique EEG IDs
    eeg_ids = df.eeg_id.unique()
    
    for eeg_id in tqdm(eeg_ids, desc="Preparing EEG data"):
        # Read EEG data from parquet file
        file_path = os.path.join(eeg_path, f"{eeg_id}.parquet")
        data = eeg_from_parquet(
            file_path, 
            eeg_features=eeg_features, 
            seq_length=config.seq_length, 
            sampling_rate=config.sampling_rate
        )
        
        # Store processed data
        all_eegs[eeg_id] = data
        
    return all_eegs


def train_fold(
    train_df, 
    valid_df, 
    train_eegs,
    config,
    fold, 
    model_dir, 
    device,
    bandpass_filter,
    rand_filter=None
):
    """
    Train model for a single fold.
    
    Args:
        train_df (pd.DataFrame): Training data for this fold
        valid_df (pd.DataFrame): Validation data for this fold
        train_eegs (dict): Dictionary of EEG data
        config (object): Model configuration
        fold (int): Fold number
        model_dir (str): Directory to save model weights
        device (torch.device): Device to train on
        bandpass_filter (dict): Bandpass filter parameters
        rand_filter (dict, optional): Random filter parameters. Defaults to None.
        
    Returns:
        tuple: (model, validation predictions, validation targets, best_loss)
    """
    # Create datasets
    train_dataset = EEGDataset(
        df=train_df,
        batch_size=config.batch_size,
        mode="train",
        eegs=train_eegs,
        bandpass_filter=bandpass_filter,
        rand_filter=rand_filter,
        config=config
    )
    
    valid_dataset = EEGDataset(
        df=valid_df,
        batch_size=config.batch_size,
        mode="valid",
        eegs=train_eegs,
        bandpass_filter=bandpass_filter,
        config=config
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # Initialize model
    model = EEGNet(
        kernels=config.kernels,
        in_channels=config.in_channels,
        fixed_kernel_size=config.fixed_kernel_size,
        num_classes=config.target_size,
        linear_layer_features=config.linear_layer_features,
    )
    
    # Move model to device
    model.to(device)
    
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    
    # Define loss function
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    # Training loop
    best_loss = float('inf')
    best_epoch = 0
    history = {
        'train_loss': [],
        'valid_loss': []
    }
    
    num_epochs = 30  # Maximum number of epochs
    patience = 5     # Early stopping patience
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Fold {fold}, Epoch {epoch+1}/{num_epochs} [Train]")):
            X = batch.pop("eeg").to(device)
            y = batch.pop("labels").to(device)
            
            optimizer.zero_grad()
            
            outputs = model(X)
            log_probs = F.log_softmax(outputs, dim=1)
            loss = criterion(log_probs, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        valid_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for step, batch in enumerate(tqdm(valid_loader, desc=f"Fold {fold}, Epoch {epoch+1}/{num_epochs} [Valid]")):
                X = batch.pop("eeg").to(device)
                y = batch.pop("labels").to(device)
                
                outputs = model(X)
                log_probs = F.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, y)
                
                valid_loss += loss.item()
                
                # Store predictions and targets
                val_preds.append(F.softmax(outputs, dim=1).cpu().numpy())
                val_targets.append(y.cpu().numpy())
        
        valid_loss /= len(valid_loader)
        history['valid_loss'].append(valid_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        
        # Check if this is the best model so far
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            
            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
            }
            
            torch.save(
                checkpoint, 
                os.path.join(model_dir, f"fold_{fold}_best.pth")
            )
            
            print(f"Saved best model for fold {fold} at epoch {epoch+1} with validation loss {best_loss:.4f}")
            
        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch+1}, no improvement for {patience} epochs")
            break
    
    # Concatenate predictions and targets
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['valid_loss'], label='Valid Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold} Training History')
    plt.legend()
    plt.savefig(os.path.join(model_dir, f"fold_{fold}_history.png"))
    plt.close()
    
    # Load best model weights
    checkpoint = torch.load(os.path.join(model_dir, f"fold_{fold}_best.pth"))
    model.load_state_dict(checkpoint["model"])
    
    return model, val_preds, val_targets, best_loss


def train_cross_validation(
    train_df,
    train_eegs,
    config,
    model_dir,
    num_folds=5,
    seed=42,
    bandpass_filter=None,
    rand_filter=None
):
    """
    Train model with cross-validation.
    
    Args:
        train_df (pd.DataFrame): Training data
        train_eegs (dict): Dictionary of EEG data
        config (object): Model configuration
        model_dir (str): Directory to save model weights
        num_folds (int, optional): Number of folds. Defaults to 5.
        seed (int, optional): Random seed. Defaults to 42.
        bandpass_filter (dict, optional): Bandpass filter parameters. Defaults to None.
        rand_filter (dict, optional): Random filter parameters. Defaults to None.
        
    Returns:
        tuple: (oof_predictions, oof_targets, scores)
    """
    # Create directory for model weights
    os.makedirs(model_dir, exist_ok=True)
    
    # Set random seed
    set_seed(seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize arrays for out-of-fold predictions
    oof_predictions = []
    oof_targets = []
    scores = []
    
    # Create folds
    gkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(gkf.split(train_df, train_df["expert_consensus"], train_df["patient_id"]))
    
    # Train on each fold
    for fold, (train_idx, valid_idx) in enumerate(splits):
        print(f"\n{'='*50}\nFold {fold+1}/{num_folds}\n{'='*50}")
        
        # Split data
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid_df = train_df.iloc[valid_idx].reset_index(drop=True)
        
        print(f"Train size: {len(fold_train_df)}, Valid size: {len(fold_valid_df)}")
        
        # Train model for this fold
        model, val_preds, val_targets, best_loss = train_fold(
            train_df=fold_train_df,
            valid_df=fold_valid_df,
            train_eegs=train_eegs,
            config=config,
            fold=fold,
            model_dir=model_dir,
            device=device,
            bandpass_filter=bandpass_filter,
            rand_filter=rand_filter
        )
        
        # Store predictions and targets
        oof_predictions.append(val_preds)
        oof_targets.append(val_targets)
        scores.append(best_loss)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Calculate overall score
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\n{'='*50}")
    print(f"Cross-validation complete!")
    print(f"Mean validation loss: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Individual fold scores: {scores}")
    print(f"{'='*50}")
    
    # Create fold ensembles
    oof_predictions = np.concatenate(oof_predictions)
    oof_targets = np.concatenate(oof_targets)
    
    # Save OOF predictions and targets
    np.save(os.path.join(model_dir, "oof_predictions.npy"), oof_predictions)
    np.save(os.path.join(model_dir, "oof_targets.npy"), oof_targets)
    
    # Save CV results
    cv_results = {
        'fold_scores': scores,
        'mean_score': float(mean_score),
        'std_score': float(std_score)
    }
    
    with open(os.path.join(model_dir, "cv_results.yaml"), 'w') as f:
        yaml.dump(cv_results, f)
    
    return oof_predictions, oof_targets, scores


def main(args):
    """
    Main training function.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create a configuration object
    class Config:
        pass
    
    config = Config()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    # Set random seed
    set_seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df, test_df = load_data(config.train_path, config.test_path)
    
    # Prepare EEG data
    print("Preparing EEG data...")
    train_eegs = prepare_eegs(
        df=train_df,
        eeg_path=config.train_eeg_path,
        eeg_features=config.eeg_features,
        config=config
    )
    
    # Define bandpass filter
    bandpass_filter = config_dict.get('bandpass_filter', {
        "low": 0.5, 
        "high": 20, 
        "order": 2
    })
    
    # Define random filter for augmentation
    rand_filter = config_dict.get('rand_filter', None)
    
    # Train model with cross-validation
    print("Starting cross-validation training...")
    model_dir = os.path.join(config.output_dir, config.model_name)
    
    oof_predictions, oof_targets, scores = train_cross_validation(
        train_df=train_df,
        train_eegs=train_eegs,
        config=config,
        model_dir=model_dir,
        num_folds=config.num_folds,
        seed=config.seed,
        bandpass_filter=bandpass_filter,
        rand_filter=rand_filter
    )
    
    print(f"Training completed. Model weights saved to {model_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet1D models for EEG classification")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    main(args)
