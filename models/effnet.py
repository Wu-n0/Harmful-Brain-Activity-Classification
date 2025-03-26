"""
EfficientNet models for HMS Brain Activity Classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import efficientnet.tfkeras as efn


class DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for EfficientNet models.
    """
    def __init__(self, data, batch_size=32, shuffle=False, mode='train', spec_size=(512, 512, 3)):
        """
        Initialize the data generator.
        
        Args:
            data (pd.DataFrame): Data to generate batches from
            batch_size (int, optional): Batch size. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle data. Defaults to False.
            mode (str, optional): 'train', 'valid', or 'test'. Defaults to 'train'.
            spec_size (tuple, optional): Size of spectrograms. Defaults to (512, 512, 3).
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.spec_size = spec_size
        self.classes = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples."""
        # Initialization
        X = np.zeros((len(indexes), *self.spec_size), dtype='float32')
        y = np.zeros((len(indexes), len(self.classes)), dtype='float32')

        # Generate data
        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            eeg_id = row['eeg_id']
            spec_offset = int(row['spectrogram_label_offset_seconds_min'])
            eeg_offset = int(row['eeg_label_offset_seconds_min'])
            file_path = f'./images/{eeg_id}_{spec_offset}_{eeg_offset}.npz'
            data = np.load(file_path)
            eeg_data = data['final_image']
            eeg_data_expanded = np.repeat(eeg_data[:, :, np.newaxis], 3, axis=2)

            X[j] = eeg_data_expanded
            if self.mode != 'test':
                y[j] = row[self.classes]

        return X, y


class DataGeneratorTest(tf.keras.utils.Sequence):
    """
    Data generator for test set using preprocessed spectrograms.
    """
    def __init__(self, data, batch_size=32, shuffle=False, eegs={}, mode='train', spec_size=(512, 512, 3)):
        """
        Initialize the test data generator.
        
        Args:
            data (pd.DataFrame): Test data
            batch_size (int, optional): Batch size. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle data. Defaults to False.
            eegs (dict, optional): Dictionary of preprocessed EEGs. Defaults to {}.
            mode (str, optional): 'train' or 'test'. Defaults to 'train'.
            spec_size (tuple, optional): Size of spectrograms. Defaults to (512, 512, 3).
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.eegs = eegs
        self.mode = mode
        self.spec_size = spec_size
        self.classes = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

    def __getitem__(self, index):
        """Generate one batch of data."""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.data))
        if self.shuffle: 
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples."""
        X = np.zeros((len(indexes), self.spec_size[0], self.spec_size[1], self.spec_size[2]), dtype='float32')
        y = np.zeros((len(indexes), 6), dtype='float32')

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            eeg_data = self.eegs[row.eeg_id] 
            eeg_data_expanded = np.repeat(eeg_data[:, :, np.newaxis], 3, axis=2)
            X[j,] = eeg_data_expanded
            if self.mode != 'test':
                y[j] = row[self.classes]

        return X, y


def learning_rate_scheduler(epoch):
    """
    Learning rate scheduler function.
    
    Args:
        epoch (int): Current epoch
        
    Returns:
        float: Learning rate for current epoch
    """
    lr_schedule = [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5]
    return lr_schedule[epoch]


def build_EfficientNetB0(input_shape=(512, 512, 3), num_classes=6):
    """
    Build an EfficientNetB0 model.
    
    Args:
        input_shape (tuple, optional): Input shape. Defaults to (512, 512, 3).
        num_classes (int, optional): Number of classes. Defaults to 6.
        
    Returns:
        tf.keras.Model: Compiled EfficientNetB0 model
    """
    # Define input
    inp = tf.keras.Input(shape=input_shape)

    # Load base model with pretrained weights
    base_model = efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)
    base_model.load_weights('./weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')

    # Build model with output layer
    x = base_model(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    # Compile model
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)

    return model
