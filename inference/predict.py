"""
Prediction script for HMS Brain Activity Classification.
This script loads all models and makes predictions on test data.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tqdm import tqdm
import argparse
from glob import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import eeg_from_parquet
from data.spectrograms import (
    generate_test_spectrograms, 
    create_custom_spectrogram, 
    preprocess_spectrogram,
    create_parquet_spectrogram
)
from models.effnet import DataGeneratorTest, build_EfficientNetB0
from models.ensemble import EnsembleModel, Config


def set_random_seed(seed=42, deterministic=True):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int, optional): Random seed. Defaults to 42.
        deterministic (bool, optional): Whether to make operations deterministic. Defaults to True.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    else:
        os.environ.pop('TF_DETERMINISTIC_OPS', None)
    

def configure_gpu():
    """
    Configure GPU settings.
    
    Returns:
        str: Device to use for computation
    """
    # Configure GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Configure memory growth for TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
                
        # Set mixed precision for TensorFlow
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print('Mixed precision enabled for TensorFlow')
    else:
        device = "cpu"
        print("CUDA not available. Using CPU")
        
    return device


def load_test_data(test_path):
    """
    Load test data.
    
    Args:
        test_path (str): Path to test CSV file
        
    Returns:
        pd.DataFrame: Test dataframe
    """
    # Load test data
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")
    
    # Add paths to test dataframe
    test_df['path_spec'] = test_df['spectrogram_id'].apply(
        lambda x: f"../input/hms-harmful-brain-activity-classification/test_spectrograms/{x}.parquet"
    )
    test_df['path_eeg'] = test_df['eeg_id'].apply(
        lambda x: f"../input/hms-harmful-brain-activity-classification/test_eegs/{x}.parquet"
    )
    
    return test_df


def prepare_eeg_data(test_df, config):
    """
    Prepare EEG data for ResNet1D model.
    
    Args:
        test_df (pd.DataFrame): Test dataframe
        config (object): Configuration object
        
    Returns:
        dict: Dictionary of EEG data
    """
    all_eegs = {}
    eeg_ids = test_df.eeg_id.unique()
    
    for eeg_id in tqdm(eeg_ids, desc="Preparing EEG data"):
        eeg_path = f"../input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet"
        data = eeg_from_parquet(eeg_path, config.eeg_features)
        all_eegs[eeg_id] = data
        
    return all_eegs


def main(args):
    """
    Main prediction function.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Set up environment
    set_random_seed(42)
    device = configure_gpu()
    
    # Load test data
    test_df = load_test_data(args.test_path)
    
    # Load configuration for ResNet1D model
    sys.path.append(args.weights_dir)
    from config import CFG as ResNetConfig
    
    # Prepare data for different models
    print("Generating spectrograms for EfficientNet models...")
    all_eegs_effnet = generate_test_spectrograms(test_df, output_folder='images')
    
    print("Preparing EEG data for ResNet1D model...")
    all_eegs_resnet = prepare_eeg_data(test_df, ResNetConfig)
    
    # Initialize ensemble model
    ensemble = EnsembleModel(device=device)
    
    # Load EfficientNet models
    print("Loading EfficientNet models...")
    ensemble.load_effnet_models(f"{args.weights_dir}/effnet")
    ensemble.load_effnet_datawide_models(f"{args.weights_dir}/effnet_datawide")
    ensemble.load_effnet_ownspec_models(f"{args.weights_dir}/effnet_ownspec")
    
    # Load ResNet1D models
    print("Loading ResNet1D models...")
    model_weights = [
        {
            'bandpass_filter': {'low': 0.5, 'high': 20, 'order': 2}, 
            'file_data': [
                {'koef': 1.0, 'file_mask': f"{args.weights_dir}/resnet1d/*_best.pth"},
            ]
        },
    ]
    ensemble.load_resnet1d_models(model_weights, ResNetConfig)
    
    # Create TensorFlow data generator
    batch_size = 32
    test_gen = DataGeneratorTest(
        test_df, 
        shuffle=False, 
        batch_size=batch_size, 
        eegs=all_eegs_effnet, 
        mode='test', 
        spec_size=(512, 512, 3)
    )
    
    # Load and initialize EfficientNet models for TensorFlow
    print("Loading TensorFlow EfficientNet models...")
    tf_models = []
    for i in range(5):  # 5-fold model
        model = build_EfficientNetB0(input_shape=(512, 512, 3), num_classes=6)
        model.load_weights(f"{args.weights_dir}/effnet_tf/MLP_fold{i}.weights.h5")
        tf_models.append(model)
    
    # Make predictions
    print("Making predictions...")
    all_preds = []
    
    for index in tqdm(test_df.index, desc="Processing samples"):
        # Get spectrogram data for current sample
        spec_path = test_df.iloc[index]['path_spec']
        eeg_path = test_df.iloc[index]['path_eeg']
        
        # Process data for EfficientNet models
        preprocessed_spec = create_parquet_spectrogram(spec_path)
        preprocessed_ownspec = create_custom_spectrogram(pd.read_parquet(eeg_path))
        
        # Make predictions with EfficientNet models
        pred_effnet = ensemble.predict_effnet(preprocessed_spec)
        pred_effnet_ownspec = ensemble.predict_effnet_ownspec(preprocessed_ownspec)
        
        # Store predictions
        all_preds.append((pred_effnet, pred_effnet_ownspec))
    
    # Make predictions with TensorFlow models
    print("Predicting with TensorFlow models...")
    tf_preds = []
    for model in tf_models:
        pred = model.predict(test_gen, verbose=0)
        tf_preds.append(pred)
    tf_preds = np.mean(tf_preds, axis=0)
    
    # Create ResNet1D PyTorch dataloader
    from torch.utils.data import DataLoader
    from models.resnet1d import EEGDataset
    
    test_dataset = EEGDataset(
        df=test_df,
        batch_size=batch_size,
        mode="test",
        eegs=all_eegs_resnet,
        bandpass_filter=model_weights[0]['bandpass_filter'],
        config=ResNetConfig
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    # Make predictions with ResNet1D models
    print("Predicting with ResNet1D models...")
    pred_resnet = ensemble.predict_resnet1d(test_loader)
    
    # Combine all predictions
    print("Combining predictions...")
    final_predictions = np.zeros((len(test_df), 6))
    for i in range(len(test_df)):
        pred_effnet, pred_effnet_ownspec = all_preds[i]
        # Weighted average of all models
        final_predictions[i] = (
            pred_resnet[i] * 0.15 + 
            tf_preds[i] * 0.50 + 
            pred_effnet * 0.175 + 
            pred_effnet_ownspec * 0.175
        )
    
    # Create submission file
    print("Creating submission file...")
    submission = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
    target_cols = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
    submission[target_cols] = final_predictions
    
    # Check that predictions sum to 1
    prediction_sums = submission[target_cols].sum(axis=1)
    print(f"Min prediction sum: {prediction_sums.min()}, Max prediction sum: {prediction_sums.max()}")
    
    # Save submission
    submission.to_csv("submission.csv", index=False)
    print(f"Submission saved to submission.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on test data")
    parser.add_argument("--test_path", type=str, default="../input/hms-harmful-brain-activity-classification/test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--weights_dir", type=str, default="./weights",
                        help="Directory containing model weights")
    args = parser.parse_args()
    
    main(args)
