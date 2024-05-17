# README for Harmful Brain Activity Classification 

This project focuses on classifying harmful brain activity using two complementary approaches: an EfficientNet-based model and a ResNet1D-GRU model. The project leverages spectrograms generated from EEG data to train these models for distinguishing between different types of brain activity.

## Key Components

### 1. Data Preparation

**Data Sources:**
- `train.csv`: Metadata for training EEG recordings.
- `train_eegs/`: Parquet files containing raw EEG data for training.
- `test.csv`: Metadata for test EEG recordings.
- `test_eegs/`: Parquet files containing raw EEG data for testing.
- `test_spectrograms/`: Parquet files containing precomputed spectrograms for testing.

**Spectrogram Generation:**
- **Training:** Generates spectrograms from raw EEG data using custom functions based on signal processing techniques (filtering, windowing, FFT).
- **Testing:** Utilizes precomputed spectrograms for efficiency.
- **Data Augmentation:** Applies label smoothing and pseudo-labeling techniques to enhance model performance.

### 2. EfficientNet Model

- **Architecture:** Employs EfficientNetB0, a pre-trained convolutional neural network, for image classification.
- **Fine-tuning:** Adapts the pre-trained model to classify different types of brain activity.
- **Loss Function:** Uses Kullback-Leibler Divergence (KLD) loss for multi-class classification.
- **Learning Rate Scheduling:** Employs a learning rate scheduler to adjust the learning rate during training.

### 3. ResNet1D-GRU Model

- **Architecture:** Combines ResNet1D blocks for feature extraction from EEG signals with GRU layers for temporal modeling.
- **Hyperparameter Tuning:** Explores different kernel sizes and filter configurations for optimal performance.
- **Model Ensembling:** Combines multiple ResNet1D-GRU models trained on different frequency bands for improved robustness.

### 4. Training and Evaluation

- **Cross-Validation:** Uses stratified group K-fold cross-validation to assess model performance and prevent overfitting.
- **Evaluation Metric:** Employs the KL divergence score, a metric commonly used in multi-class classification tasks.

### 5. Inference and Submission

- **Inference on Test Data:** Generates predictions for the test spectrograms using both models.
- **Ensemble Prediction:** Combines the predictions of the EfficientNet and ResNet1D-GRU models using a weighted average.
- **Submission:** Formats the final predictions into a CSV file for submission.

## Additional Notes

- **Mixed Precision:** Utilizes mixed precision training to accelerate computations on GPUs.
- **Deterministic Behavior:** Sets random seeds to ensure reproducibility of results.

Feel free to explore and modify the code to experiment with different architectures, hyperparameters, and data preprocessing techniques to potentially improve classification performance.
