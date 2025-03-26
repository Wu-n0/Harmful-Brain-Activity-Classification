"""
Training script for EfficientNet models for HMS Brain Activity Classification.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import gc

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.effnet import build_EfficientNetB0, DataGenerator, learning_rate_scheduler
from data.preprocessing import create_train_data


def set_random_seed(seed=42, deterministic=True):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int, optional): Random seed. Defaults to 42.
        deterministic (bool, optional): Whether to use deterministic operations. Defaults to True.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    else:
        os.environ.pop('TF_DETERMINISTIC_OPS', None)


def configure_gpu():
    """
    Configure GPU settings for TensorFlow.
    """
    # Set the visible CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Set the strategy for using GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f'Using {len(gpus)} GPU')
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Using {len(gpus)} GPUs')

    # Configure memory growth
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Enable or disable mixed precision
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print('Mixed precision enabled')


def train_model(
    train_data, 
    train_data_2, 
    folds, 
    random_seed, 
    targets, 
    model_name,
    output_dir="./models",
    batch_size=32,
    epochs=4,
    spec_size=(512, 512, 3)
):
    """
    Train EfficientNet model with label refinement strategy.
    
    Args:
        train_data (pd.DataFrame): High-quality training data
        train_data_2 (pd.DataFrame): Low-quality training data
        folds (int): Number of cross-validation folds
        random_seed (int): Random seed for reproducibility
        targets (list): Target column names
        model_name (str): Name of the model
        output_dir (str, optional): Output directory for model weights. Defaults to "./models".
        batch_size (int, optional): Batch size. Defaults to 32.
        epochs (int, optional): Number of epochs. Defaults to 4.
        spec_size (tuple, optional): Size of spectrograms. Defaults to (512, 512, 3).
        
    Returns:
        tuple: (cv_score, training_time, oof_predictions, train_mean_oof, true_labels, models, scores, std_dev, model_path)
    """
    import time
    from kaggle_kl_div import score
    
    start_time = time.time()
    model_path = f'{output_dir}/{model_name}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    all_oof = []
    all_oof2 = []
    all_true = []
    models = []
    score_list = []
    
    # Create stratified folds
    gkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    splits1 = list(gkf.split(train_data, train_data[["expert_consensus"]], train_data["patient_id"]))
    splits2 = list(gkf.split(train_data_2, train_data_2[["expert_consensus"]], train_data_2["patient_id"]))

    # Define learning rate scheduler callback
    LR = LearningRateScheduler(learning_rate_scheduler, verbose=True)

    # Iterate over folds
    for i, ((train_index, valid_index), (train_index2, valid_index2)) in enumerate(zip(splits1, splits2)):
        
        # Copy dataframes to avoid leaks
        train_data_ = train_data.copy()
        train_data_2_ = train_data_2.copy()
        set_random_seed(random_seed, deterministic=True)
        
        # Start folding
        print('#' * 25)
        print(f'### Fold {i + 1}')
        print(f'### train size 1 {len(train_index)}, valid size {len(valid_index)}')
        print(f'### train size 2 {len(train_index2)}, valid size {len(valid_index2)}')
        print('#' * 25)

        ### --------------------------- Performs model 1 training -------------- --------------------------- ###
        K.clear_session()
        train_gen = DataGenerator(train_data_.iloc[train_index], shuffle=True, batch_size=batch_size, spec_size=spec_size)
        valid_gen = DataGenerator(
            train_data_.iloc[valid_index], 
            shuffle=False, 
            batch_size=(batch_size*2), 
            mode='valid',
            spec_size=spec_size
        )
        
        # Build and train initial model
        model = build_EfficientNetB0(input_shape=spec_size, num_classes=len(targets))
        
        # Add callbacks
        checkpoint = ModelCheckpoint(
            f'{model_path}/model1_fold{i}.weights.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_weights_only=True
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=3,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        
        # Train first model
        history = model.fit(
            train_gen, 
            verbose=2, 
            validation_data=valid_gen, 
            epochs=epochs, 
            callbacks=[LR, checkpoint, early_stopping]
        )

        # Model training result 1
        train_loss = history.history['loss'][-1]  
        valid_loss = history.history['val_loss'][-1]
        print(f'train_loss 1 {train_loss} valid_loss 1 {valid_loss}')
        score_list.append((train_loss, valid_loss))

        
        ### --------------------------- Creation of pseudo labels ---------------- ------------------------- ###
        # Pseudo labels for low quality data
        train_2_index_total_gen = DataGenerator(
            train_data_2_.iloc[train_index2], 
            shuffle=False, 
            batch_size=batch_size,
            spec_size=spec_size
        )
        pseudo_labels_2 = model.predict(train_2_index_total_gen, verbose=2)
        
        # Refinement of low quality labels (50% original, 50% predicted)
        train_data_2_.loc[train_index2, targets] /= 2
        train_data_2_.loc[train_index2, targets] += pseudo_labels_2 / 2

        # Pseudo labels for high quality data (50% of data)
        train_data_3_ = train_data_.copy()
        train_3_index_total_gen = DataGenerator(
            train_data_3_.iloc[train_index], 
            shuffle=False, 
            batch_size=batch_size,
            spec_size=spec_size
        )
        pseudo_labels_3 = model.predict(train_3_index_total_gen, verbose=2)
        
        # Refinement of high quality labels (50% original, 50% predicted)
        train_data_3_.loc[train_index, targets] /= 2
        train_data_3_.loc[train_index, targets] += pseudo_labels_3 / 2

        ### --------------------------- Creation of the data generator for the refined labels model --------- ###
        # Use a subset of high quality data and all low quality data
        np.random.shuffle(train_index)
        np.random.shuffle(valid_index)
        sixty_percent_length = int(0.5 * len(train_data_3_))
        train_index_60 = train_index[:int(sixty_percent_length * len(train_index) / len(train_data_3_))]
        valid_index_60 = valid_index[:int(sixty_percent_length * len(valid_index) / len(train_data_3_))]
        
        # Create data generator with refined labels
        train_gen_2 = DataGenerator(
            pd.concat([train_data_3_.iloc[train_index_60], train_data_2_.iloc[train_index2]]), 
            shuffle=True, 
            batch_size=batch_size,
            spec_size=spec_size
        )
        valid_gen_2 = DataGenerator(
            pd.concat([train_data_3_.iloc[valid_index_60], train_data_2_.iloc[valid_index2]]), 
            shuffle=False, 
            batch_size=batch_size*2, 
            mode='valid',
            spec_size=spec_size
        )
        
        # Rebuild the high quality data generator with original labels
        train_gen = DataGenerator(
            train_data_.iloc[train_index], 
            shuffle=True, 
            batch_size=batch_size,
            spec_size=spec_size
        )
        valid_gen = DataGenerator(
            train_data_.iloc[valid_index], 
            shuffle=False, 
            batch_size=(batch_size*2), 
            mode='valid',
            spec_size=spec_size
        )
        
        ### --------------------------- Model 2 training and finetuning -------------- ###
        K.clear_session()
        new_model = build_EfficientNetB0(input_shape=spec_size, num_classes=len(targets))
        
        # Add callbacks
        checkpoint2 = ModelCheckpoint(
            f'{model_path}/model2_fold{i}.weights.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_weights_only=True
        )
        
        # Training with the refined low-quality data
        history = new_model.fit(
            train_gen_2, 
            verbose=2, 
            validation_data=valid_gen_2, 
            epochs=epochs, 
            callbacks=[LR, checkpoint2]
        )
        
        # Finetuning with original high-quality data
        history = new_model.fit(
            train_gen, 
            verbose=2, 
            validation_data=valid_gen, 
            epochs=epochs, 
            callbacks=[LR]
        )
        
        # Save final model weights
        new_model.save_weights(f'{model_path}/MLP_fold{i}.weights.h5')
        models.append(new_model)

        # Model 2 training result
        train_loss = history.history['loss'][-1]
        valid_loss = history.history['val_loss'][-1]
        print(f'train_loss 2 {train_loss} valid_loss 2 {valid_loss}')
        score_list.append((train_loss, valid_loss))

        # Calculate out-of-fold predictions
        oof = new_model.predict(valid_gen, verbose=2)
        all_oof.append(oof)
        all_true.append(train_data.iloc[valid_index][targets].values)

        # Calculate train mean OOF (baseline)
        y_train = train_data.iloc[train_index][targets].values
        y_valid = train_data.iloc[valid_index][targets].values
        oof = y_valid.copy()
        for j in range(len(targets)):
            oof[:,j] = y_train[:,j].mean()
        oof = oof / oof.sum(axis=1, keepdims=True)
        all_oof2.append(oof)

        # Clean up
        del model, new_model, train_gen, valid_gen, train_2_index_total_gen, train_gen_2, valid_gen_2, oof, y_train, y_valid
        K.clear_session()
        gc.collect()

        if i == folds-1:
            break

    # Concatenate all OOF predictions
    all_oof = np.concatenate(all_oof)
    all_oof2 = np.concatenate(all_oof2)
    all_true = np.concatenate(all_true)

    # Calculate cross-validation score
    oof = pd.DataFrame(all_oof.copy())
    oof['id'] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true['id'] = np.arange(len(true))

    cv = score(solution=true, submission=oof, row_id_column_name='id')
    end_time = time.time()
    training_time = end_time - start_time
    print(f'{model_name} CV Score with EEG Spectrograms = {cv}, training time: {training_time:.2f}s')
    
    # Calculate standard deviation of scores
    score_array = np.array(score_list)
    std_dev = np.std(score_array, axis=0)
    std_dev = std_dev.tolist()

    # Plot training metrics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot([x[0] for x in score_list[::2]], label='Train Loss (Model 1)')
    plt.plot([x[0] for x in score_list[1::2]], label='Train Loss (Model 2)')
    plt.title('Train Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot([x[1] for x in score_list[::2]], label='Validation Loss (Model 1)')
    plt.plot([x[1] for x in score_list[1::2]], label='Validation Loss (Model 2)')
    plt.title('Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_path}/training_metrics.png')
    plt.close()

    return cv, training_time, all_oof, all_oof2, all_true, models, score_list, std_dev, model_path


def main(args):
    """
    Main training function.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Create a configuration object
    class Config:
        pass
    
    config = Config()
    
    # Load configuration
    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
        # Load YAML configuration
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Set attributes from dictionary
        for key, value in config_dict.items():
            setattr(config, key, value)
    else:
        # Load Python configuration
        sys.path.append(os.path.dirname(args.config))
        config_module = __import__(os.path.basename(args.config).replace('.py', ''))
        
        # Use the existing Config class
        if hasattr(config_module, 'CFG'):
            config = config_module.CFG
    
    # Set up environment
    set_random_seed(seed=config.get('seed', 42))
    configure_gpu()
    
    # Create output directory
    output_dir = config.get('output_dir', './models')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create training data
    print("Creating training datasets...")
    high_quality_df, low_quality_df = create_train_data()
    
    # Train model
    print("Starting model training...")
    result = train_model(
        train_data=high_quality_df,
        train_data_2=low_quality_df,
        folds=config.get('folds', 5),
        random_seed=config.get('seed', 42),
        targets=config.get('target_cols', ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]),
        model_name=config.get('model_name', 'effnet_model'),
        output_dir=output_dir,
        batch_size=config.get('batch_size', 32),
        epochs=config.get('epochs', 4),
        spec_size=tuple(config.get('spec_size', (512, 512, 3)))
    )
    
    cv_score = result[0]
    
    # Save training results
    result_data = {
        'cv_score': float(cv_score),
        'training_time': float(result[1]),
        'score_list': [[float(x) for x in sublist] for sublist in result[6]],
        'std_dev': [float(x) for x in result[7]]
    }
    
    with open(f'{output_dir}/{config["model_name"]}/training_results.yaml', 'w') as f:
        yaml.dump(result_data, f)
    
    print(f"Training completed. Final CV score: {cv_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet1D models for EEG classification")
    parser.add_argument("--config", type=str, default="../config/config.py", 
                       help="Path to configuration file (YAML or Python)")
    args = parser.parse_args()
    
    main(args)
