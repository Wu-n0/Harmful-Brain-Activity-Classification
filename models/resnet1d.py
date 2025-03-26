"""
ResNet1D with GRU model for HMS Brain Activity Classification.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """
    Dataset class for EEG data.
    """
    def __init__(
        self,
        df,
        batch_size,
        eegs,
        mode="train",
        downsample=None,
        bandpass_filter=None,
        rand_filter=None,
        config=None
    ):
        """
        Initialize the EEG dataset.
        
        Args:
            df (pd.DataFrame): Data to generate batches from
            batch_size (int): Batch size
            eegs (dict): Dictionary of EEG data
            mode (str, optional): 'train', 'valid', or 'test'. Defaults to 'train'.
            downsample (int, optional): Downsample factor. Defaults to None.
            bandpass_filter (dict, optional): Bandpass filter parameters. Defaults to None.
            rand_filter (dict, optional): Random filter parameters. Defaults to None.
            config (object, optional): Configuration object. Defaults to None.
        """
        self.df = df
        self.batch_size = batch_size
        self.mode = mode
        self.eegs = eegs
        self.downsample = downsample
        self.bandpass_filter = bandpass_filter
        self.rand_filter = rand_filter
        self.config = config
        
    def __len__(self):
        """Length of dataset."""
        return len(self.df)

    def __getitem__(self, index):
        """Get one item."""
        X, y_prob = self.__data_generation(index)
        if self.downsample is not None:
            X = X[:: self.downsample, :]
        output = {
            "eeg": torch.tensor(X, dtype=torch.float32),
            "labels": torch.tensor(y_prob, dtype=torch.float32),
        }
        return output

    def __data_generation(self, index):
        """Generate data for a single sample."""
        from data.preprocessing import butter_bandpass_filter, butter_lowpass_filter
        
        # Initialize with zeros
        X = np.zeros(
            (self.config.out_samples, self.config.in_channels), dtype="float32"
        )

        # Get row and corresponding EEG data
        row = self.df.iloc[index]
        data = self.eegs[row.eeg_id]
        
        # Get middle portion if needed
        if self.config.nsamples != self.config.out_samples:
            if self.mode != "train":
                offset = (self.config.nsamples - self.config.out_samples) // 2
            else:
                offset = ((self.config.nsamples - self.config.out_samples) * random.randint(0, 1000)) // 1000
            data = data[offset:offset+self.config.out_samples, :]

        # Process each electrode pair
        for i, (feat_a, feat_b) in enumerate(self.config.map_features):
            if self.mode == "train" and self.config.random_close_zone > 0 and random.uniform(0.0, 1.0) <= self.config.random_close_zone:
                continue
                
            # Compute differential signals
            diff_feat = (
                data[:, self.config.feature_to_index[feat_a]]
                - data[:, self.config.feature_to_index[feat_b]]
            )

            # Apply bandpass filter if specified
            if self.bandpass_filter is not None:
                diff_feat = butter_bandpass_filter(
                    diff_feat,
                    self.bandpass_filter["low"],
                    self.bandpass_filter["high"],
                    self.config.sampling_rate,
                    order=self.bandpass_filter["order"],
                )
                    
            # Apply random filter if in training mode and specified
            if (
                self.mode == "train"
                and self.rand_filter is not None
                and random.uniform(0.0, 1.0) <= self.rand_filter["probab"]
            ):
                lowcut = random.randint(
                    self.rand_filter["low"], self.rand_filter["high"]
                )
                highcut = lowcut + self.rand_filter["band"]
                diff_feat = butter_bandpass_filter(
                    diff_feat,
                    lowcut,
                    highcut,
                    self.config.sampling_rate,
                    order=self.rand_filter["order"],
                )

            X[:, i] = diff_feat

        # Process frequency channels if specified
        n = self.config.n_map_features
        if len(self.config.freq_channels) > 0:
            for i in range(self.config.n_map_features):
                diff_feat = X[:, i]
                for j, (lowcut, highcut) in enumerate(self.config.freq_channels):
                    band_feat = butter_bandpass_filter(
                        diff_feat, lowcut, highcut, self.config.sampling_rate, 
                        order=self.config.filter_order,
                    )
                    X[:, n] = band_feat
                    n += 1

        # Process simple features if specified
        for spml_feat in self.config.simple_features:
            feat_val = data[:, self.config.feature_to_index[spml_feat]]
            
            if self.bandpass_filter is not None:
                feat_val = butter_bandpass_filter(
                    feat_val,
                    self.bandpass_filter["low"],
                    self.bandpass_filter["high"],
                    self.config.sampling_rate,
                    order=self.bandpass_filter["order"],
                )

            if (
                self.mode == "train"
                and self.rand_filter is not None
                and random.uniform(0.0, 1.0) <= self.rand_filter["probab"]
            ):
                lowcut = random.randint(
                    self.rand_filter["low"], self.rand_filter["high"]
                )
                highcut = lowcut + self.rand_filter["band"]
                feat_val = butter_bandpass_filter(
                    feat_val,
                    lowcut,
                    highcut,
                    self.config.sampling_rate,
                    order=self.rand_filter["order"],
                )

            X[:, n] = feat_val
            n += 1
            
        # Clip to reasonable range
        X = np.clip(X, -1024, 1024)

        # Replace NaN with zero and scale
        X = np.nan_to_num(X, nan=0) / 32.0

        # Apply lowpass filter
        X = butter_lowpass_filter(X, order=self.config.filter_order)

        # Prepare target
        y_prob = np.zeros(self.config.target_size, dtype="float32")
        if self.mode != "test":
            y_prob = row[self.config.target_cols].values.astype(np.float32)

        return X, y_prob


class ResNet_1D_Block(nn.Module):
    """
    ResNet 1D block with residual connections.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        downsampling,
        dilation=1,
        groups=1,
        dropout=0.0,
    ):
        """
        Initialize the ResNet 1D block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size
            stride (int): Stride
            padding (int): Padding
            downsampling (nn.Module): Downsampling module
            dilation (int, optional): Dilation. Defaults to 1.
            groups (int, optional): Groups. Defaults to 1.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(ResNet_1D_Block, self).__init__()

        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu_1 = nn.Hardswish()
        self.relu_2 = nn.Hardswish()

        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=dilation,
        )
        self.downsampling = downsampling

    def forward(self, x):
        """Forward pass."""
        identity = x

        out = self.bn1(x)
        out = self.relu_1(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


class EEGNet(nn.Module):
    """
    EEGNet model with ResNet1D blocks and GRU for temporal modeling.
    """
    def __init__(
        self,
        kernels,
        in_channels,
        fixed_kernel_size,
        num_classes,
        linear_layer_features,
        dilation=1,
        groups=1,
    ):
        """
        Initialize the EEGNet model.
        
        Args:
            kernels (list): List of kernel sizes for parallel convolutions
            in_channels (int): Number of input channels
            fixed_kernel_size (int): Fixed kernel size for later layers
            num_classes (int): Number of output classes
            linear_layer_features (int): Number of features in linear layer
            dilation (int, optional): Dilation. Defaults to 1.
            groups (int, optional): Groups. Defaults to 1.
        """
        super(EEGNet, self).__init__()
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels

        # Create parallel convolutions with different kernel sizes
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=(kernel_size),
                stride=1,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.parallel_conv.append(sep_conv)

        # Initial layers
        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu_1 = nn.SiLU()
        self.relu_2 = nn.SiLU()

        self.conv1 = nn.Conv1d(
            in_channels=self.planes,
            out_channels=self.planes,
            kernel_size=fixed_kernel_size,
            stride=2,
            padding=2,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        # ResNet blocks
        self.block = self._make_resnet_layer(
            kernel_size=fixed_kernel_size,
            stride=1,
            dilation=dilation,
            groups=groups,
            padding=fixed_kernel_size // 2,
        )
        
        # Final layers
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)

        # GRU for temporal modeling
        self.rnn = nn.GRU(
            input_size=self.in_channels,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
        )

        # Final classification layer
        self.fc = nn.Linear(in_features=linear_layer_features, out_features=num_classes)

    def _make_resnet_layer(
        self,
        kernel_size,
        stride,
        dilation=1,
        groups=1,
        blocks=9,
        padding=0,
        dropout=0.0,
    ):
        """Create a layer of ResNet blocks."""
        layers = []
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            )
            layers.append(
                ResNet_1D_Block(
                    in_channels=self.planes,
                    out_channels=self.planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling,
                    dilation=dilation,
                    groups=groups,
                    dropout=dropout,
                )
            )
        return nn.Sequential(*layers)

    def extract_features(self, x):
        """Extract features from input."""
        x = x.permute(0, 2, 1)
        out_sep = []

        # Apply parallel convolutions
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu_1(out)
        out = self.conv1(out)

        # Apply ResNet blocks
        out = self.block(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.avgpool(out)

        # Flatten for linear layer
        out = out.reshape(out.shape[0], -1)
        
        # Apply GRU
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]

        # Concatenate CNN and RNN features
        new_out = torch.cat([out, new_rnn_h], dim=1)
        return new_out

    def forward(self, x):
        """Forward pass."""
        new_out = self.extract_features(x)
        result = self.fc(new_out)
        return result


def inference_function(test_loader, model, device):
    """
    Run inference on test data.
    
    Args:
        test_loader (DataLoader): Test data loader
        model (nn.Module): Model to use for inference
        device (torch.device): Device to run inference on
        
    Returns:
        dict: Dictionary with predictions
    """
    model.eval()  # set model in evaluation mode
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    
    for step, batch in enumerate(test_loader):
        X = batch.pop("eeg").to(device)  # send inputs to device
        with torch.no_grad():
            y_preds = model(X)  # forward propagation pass
        y_preds = softmax(y_preds)
        preds.append(y_preds.to("cpu").numpy())  # save predictions

    prediction_dict["predictions"] = np.concatenate(preds)
    return prediction_dict
