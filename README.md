# HMS Harmful Brain Activity Classification

![Brain EEG](https://img.shields.io/badge/EEG-Analysis-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Classification-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20TensorFlow-orange)

## Project Overview

This repository contains my solution to the [HMS Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) Kaggle competition. The goal was to detect and classify seizures and other types of harmful brain activity from electroencephalography (EEG) signals recorded from critically ill hospital patients.

### Challenge

The competition involved classifying EEG data into six patterns:
- Seizure (SZ)
- Generalized Periodic Discharges (GPD)
- Lateralized Periodic Discharges (LPD)
- Lateralized Rhythmic Delta Activity (LRDA)
- Generalized Rhythmic Delta Activity (GRDA)
- Other

The primary challenge was that even experts have difficulty classifying these patterns and often disagree on the correct labels. The task was to predict probabilities of each class that match the distribution of expert annotations.

## Solution Approach

My solution used an ensemble of three different models:

### 1. EfficientNetB0 with Custom Spectrograms
- Used EEG data to generate custom spectrograms with advanced signal processing 
- Implemented a label refinement strategy to improve classification quality
- Applied data augmentation techniques for better generalization

### 2. ResNet1D with GRU for Raw Signal Processing
- Built a custom 1D CNN architecture with residual connections
- Added recurrent layers (GRU) to capture temporal dependencies
- Utilized multi-kernel approach to capture features at different scales

### 3. Ensemble of EfficientNet Models with Different Spectrograms
- Combined multiple models including EfficientNetB0 and EfficientNetB1
- Used both instance-wise and dataset-wide normalization
- Created new spectrograms directly from EEG data for additional input features

### Ensemble Weighting
The final solution combines predictions from all three models with weights:
- ResNet1D_GRU: 15%
- EfficientNetB0 with Custom Spectrograms: 50%
- EfficientNet Ensemble: 35%

## Repository Structure

```
.
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
├── config/                   # Configuration files
├── data/                     # Data preprocessing scripts
│   ├── preprocessing.py
│   └── spectrograms.py
├── models/                   # Model definitions
│   ├── effnet.py             # EfficientNet implementation
│   ├── resnet1d.py           # ResNet1D with GRU
│   └── ensemble.py           # Model ensemble
├── training/                 # Training scripts
│   ├── train_effnet.py       # EfficientNet training
│   └── train_resnet1d.py     # ResNet1D training
└── inference/                # Inference and submission scripts
    └── predict.py            # Final prediction script
```

## Performance

This solution achieved a competitive Kullback-Leibler divergence score in the competition, demonstrating the effectiveness of the ensemble approach in matching expert consensus on challenging EEG pattern classification tasks.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hms-brain-activity.git
cd hms-brain-activity

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Inference
```bash
python inference/predict.py --test_path /path/to/test_data --weights_dir /path/to/weights
```

### Training (Optional)
```bash
python training/train_effnet.py --config config/effnet_config.yaml
python training/train_resnet1d.py --config config/resnet1d_config.yaml
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
