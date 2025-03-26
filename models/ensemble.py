"""
Ensemble model combining multiple models for HMS Brain Activity Classification.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from glob import glob
from tqdm.auto import tqdm
import tensorflow as tf

from models.resnet1d import EEGNet, inference_function


class Config:
    """
    Configuration class for the ensemble model.
    """
    seed = 3131
    image_transform = None  # Will be set in __init__
    num_folds = 5
    dataset_wide_mean = -0.2972692229201065  # From Train notebook
    dataset_wide_std = 2.5997336315611026    # From Train notebook
    ownspec_mean = 7.29084372799223e-05      # From Train spectrograms notebook
    ownspec_std = 4.510082606216031          # From Train spectrograms notebook
    
    def __init__(self):
        """Initialize with torch transforms."""
        import torchvision.transforms as transforms
        self.image_transform = transforms.Resize((512, 512))


class EnsembleModel:
    """
    Ensemble model combining multiple models.
    """
    def __init__(self, config=None, device='cuda'):
        """
        Initialize the ensemble model.
        
        Args:
            config (Config, optional): Configuration object. Defaults to None.
            device (str, optional): Device to run model on. Defaults to 'cuda'.
        """
        self.config = config or Config()
        self.device = torch.device(device)
        self.effnet_models = []
        self.effnet_datawide_models = []
        self.effnet_ownspec_models = []
        self.resnet1d_models = []
        self.weight_effnet = 0.15
        self.weight_resnet1d = 0.50
        self.weight_effnet_datawide = 0.35
        
    def load_effnet_models(self, model_dir='./weights/effnet'):
        """
        Load EfficientNet models.
        
        Args:
            model_dir (str, optional): Directory with model weights. Defaults to './weights/effnet'.
        """
        # Load original EfficientNetB0 models
        for i in range(self.config.num_folds):
            model_effnet_b0 = timm.create_model('efficientnet_b0', pretrained=False, num_classes=6, in_chans=1)
            model_effnet_b0.load_state_dict(torch.load(f'{model_dir}/efficientnet_b0_fold{i}.pth', map_location=self.device))
            model_effnet_b0.to(self.device)
            model_effnet_b0.eval()
            self.effnet_models.append(model_effnet_b0)
            
    def load_effnet_datawide_models(self, model_dir='./weights/effnet_datawide'):
        """
        Load EfficientNet models trained with dataset-wide normalization.
        
        Args:
            model_dir (str, optional): Directory with model weights. Defaults to './weights/effnet_datawide'.
        """
        # Load hyperparameter optimized EfficientNetB1
        for i in range(self.config.num_folds):
            model_effnet_b1 = timm.create_model('efficientnet_b1', pretrained=False, num_classes=6, in_chans=1)
            model_effnet_b1.load_state_dict(torch.load(f'{model_dir}/efficientnet_b1_fold{i}.pth', map_location=self.device))
            model_effnet_b1.to(self.device)
            model_effnet_b1.eval()
            self.effnet_datawide_models.append(model_effnet_b1)
            
    def load_effnet_ownspec_models(self, model_dir='./weights/effnet_ownspec'):
        """
        Load EfficientNet models trained with custom spectrograms.
        
        Args:
            model_dir (str, optional): Directory with model weights. Defaults to './weights/effnet_ownspec'.
        """
        # Load EfficientNetB1 with new spectrograms
        for i in range(self.config.num_folds):
            model_effnet_b1 = timm.create_model('efficientnet_b1', pretrained=False, num_classes=6, in_chans=1)
            model_effnet_b1.load_state_dict(torch.load(
                f'{model_dir}/efficientnet_b1_fold{i}_datawide_CosineAnnealingLR_0.001_False.pth', 
                map_location=self.device
            ))
            model_effnet_b1.to(self.device)
            model_effnet_b1.eval()
            self.effnet_ownspec_models.append(model_effnet_b1)
    
    def load_resnet1d_models(self, model_config, resnet_config=None):
        """
        Load ResNet1D models.
        
        Args:
            model_config (dict): Model configuration
            resnet_config (object, optional): ResNet configuration. Defaults to None.
        """
        koef_sum = 0
        koef_count = 0
        
        for model_block in model_config:
            # Create model
            model = EEGNet(
                kernels=resnet_config.kernels,
                in_channels=resnet_config.in_channels,
                fixed_kernel_size=resnet_config.fixed_kernel_size,
                num_classes=resnet_config.target_size,
                linear_layer_features=resnet_config.linear_layer_features,
            )
            
            # Load weights
            for file_line in model_block['file_data']:
                koef = file_line['koef']
                for weight_model_file in glob(file_line['file_mask']):
                    checkpoint = torch.load(weight_model_file, map_location=self.device)
                    model.load_state_dict(checkpoint["model"])
                    model.to(self.device)
                    model.eval()
                    self.resnet1d_models.append((model, koef))
                    koef_sum += koef
                    koef_count += 1
                    
        # Normalize coefficients
        if koef_count > 0:
            koef_norm = koef_sum / koef_count
            self.resnet1d_models = [(model, koef/koef_norm) for model, koef in self.resnet1d_models]
    
    def load_tensorflow_models(self, model_dirs, num_folds=5):
        """
        Load TensorFlow models.
        
        Args:
            model_dirs (list): List of model directories
            num_folds (int, optional): Number of folds. Defaults to 5.
        """
        self.tf_models = []
        for model_dir in model_dirs:
            for i in range(num_folds):
                model = tf.keras.models.load_model(f'{model_dir}/fold_{i}')
                self.tf_models.append(model)
    
    def preprocess_ownspec(self, data):
        """
        Preprocess data for custom spectrogram models.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            torch.Tensor: Processed data tensor
        """
        # Handle NaN values
        mask = np.isnan(data)
        data[mask] = -1
        
        # Clip and log transform
        data = np.clip(data, np.exp(-6), np.exp(10))
        data = np.log(data)
        
        # Normalize
        eps = 1e-6
        data = (data - self.config.ownspec_mean) / (self.config.ownspec_std + eps)
        
        # Convert to tensor and resize
        data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
        data = self.config.image_transform(data_tensor)
        
        return data
    
    def preprocess_datawide(self, data):
        """
        Preprocess data using dataset-wide normalization.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            torch.Tensor: Processed data tensor
        """
        eps = 1e-6
        data = (data - self.config.dataset_wide_mean) / (self.config.dataset_wide_std + eps)
        data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
        data = self.config.image_transform(data_tensor)
        return data
    
    def preprocess_instance_wise(self, data):
        """
        Preprocess data using instance-wise normalization.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            torch.Tensor: Processed data tensor
        """
        eps = 1e-6
        data_mean = data.mean(axis=(0, 1))
        data_std = data.std(axis=(0, 1))
        data = (data - data_mean) / (data_std + eps)
        
        data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
        data = self.config.image_transform(data_tensor)
        
        return data
    
    def predict_effnet(self, preprocessed_data):
        """
        Predict using EfficientNet models.
        
        Args:
            preprocessed_data (np.ndarray): Preprocessed data
            
        Returns:
            np.ndarray: Model predictions
        """
        predictions = []
        
        # Predict with original EfficientNet models
        for model in self.effnet_models:
            data = self.preprocess_instance_wise(preprocessed_data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = model(data)
                pred = F.softmax(output)[0].detach().cpu().numpy()
            predictions.append(pred)
            
        # Predict with dataset-wide normalized models
        for model in self.effnet_datawide_models:
            data = self.preprocess_datawide(preprocessed_data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = model(data)
                pred = F.softmax(output)[0].detach().cpu().numpy()
            predictions.append(pred)
            
        return np.mean(predictions, axis=0) if predictions else None
    
    def predict_effnet_ownspec(self, preprocessed_data):
        """
        Predict using EfficientNet models with custom spectrograms.
        
        Args:
            preprocessed_data (np.ndarray): Preprocessed data
            
        Returns:
            np.ndarray: Model predictions
        """
        predictions = []
        
        # Predict with custom spectrogram models
        for model in self.effnet_ownspec_models:
            data = self.preprocess_ownspec(preprocessed_data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = model(data)
                pred = F.softmax(output)[0].detach().cpu().numpy()
            predictions.append(pred)
            
        return np.mean(predictions, axis=0) if predictions else None
    
    def predict_resnet1d(self, test_loader):
        """
        Predict using ResNet1D models.
        
        Args:
            test_loader (DataLoader): Test data loader
            
        Returns:
            np.ndarray: Model predictions
        """
        predictions = []
        
        for model, koef in self.resnet1d_models:
            prediction_dict = inference_function(test_loader, model, self.device)
            pred = prediction_dict["predictions"] * koef
            predictions.append(pred)
            
        return np.mean(predictions, axis=0) if predictions else None
    
    def predict_tensorflow(self, data_generator):
        """
        Predict using TensorFlow models.
        
        Args:
            data_generator (DataGenerator): Data generator
            
        Returns:
            np.ndarray: Model predictions
        """
        predictions = []
        
        for model in self.tf_models:
            pred = model.predict(data_generator, verbose=0)
            predictions.append(pred)
            
        return np.mean(predictions, axis=0) if predictions else None
    
    def ensemble_predictions(self, pred1, pred2, pred3=None):
        """
        Combine predictions from multiple models.
        
        Args:
            pred1 (np.ndarray): Predictions from first model
            pred2 (np.ndarray): Predictions from second model
            pred3 (np.ndarray, optional): Predictions from third model. Defaults to None.
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if pred3 is not None:
            return (
                pred1 * self.weight_resnet1d + 
                pred2 * self.weight_effnet + 
                pred3 * self.weight_effnet_datawide
            )
        else:
            # Adjust weights if only two models
            w1 = self.weight_resnet1d / (self.weight_resnet1d + self.weight_effnet)
            w2 = self.weight_effnet / (self.weight_resnet1d + self.weight_effnet)
            return pred1 * w1 + pred2 * w2
