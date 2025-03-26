"""
Spectrogram generation utilities for HMS Brain Activity Classification.
This module contains functions for creating different types of spectrograms from EEG data.
"""

import numpy as np
import pandas as pd
import cv2
from scipy import signal
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_test_spectrograms(test_df, output_folder='images'):
    """
    Generate spectrograms for test data.
    
    Args:
        test_df (pd.DataFrame): Test dataframe
        output_folder (str, optional): Output folder for spectrograms. Defaults to 'images'.
    
    Returns:
        dict: Dictionary mapping EEG IDs to spectrograms
    """
    # Import from the same package
    from data.preprocessing import create_spectrogram_with_cusignal, create_spectrogram_competition
    
    # Make sure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    all_eegs = {}
    
    # Process each EEG in the test set
    for i in tqdm(range(len(test_df)), desc="Processing EEGs"):
        row = test_df.iloc[i]
        eeg_id = row['eeg_id']
        spec_id = row['spectrogram_id']
        seconds_min = 0
        start_second = 0
        
        # Read EEG data
        eeg_data = pd.read_parquet(f'../input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet')
        eeg_new_key = eeg_id
        
        # Create 50-second spectrogram
        image_50s, _ = create_spectrogram_with_cusignal(
            eeg_data=eeg_data, 
            eeg_id=eeg_id, 
            start=start_second, 
            duration=50,
            low_cut_freq=0.7, 
            high_cut_freq=20, 
            order_band=5,
            spec_size_freq=267, 
            spec_size_time=501,
            nperseg_=1500, 
            noverlap_=1483, 
            nfft_=2750,
            sigma_gaussian=0.0, 
            mean_montage_names=4
        )
        
        # Create 10-second spectrogram
        image_10s, _ = create_spectrogram_with_cusignal(
            eeg_data=eeg_data, 
            eeg_id=eeg_id, 
            start=start_second, 
            duration=10,
            low_cut_freq=0.7, 
            high_cut_freq=20, 
            order_band=5,
            spec_size_freq=100, 
            spec_size_time=291,
            nperseg_=260, 
            noverlap_=254, 
            nfft_=1030,
            sigma_gaussian=0.0, 
            mean_montage_names=4
        )
        
        # Create competition-format spectrogram
        image_10m = create_spectrogram_competition(spec_id, seconds_min)
        
        # Combine different spectrograms
        imagem_final_unico_canal = np.zeros((1068, 501))
        for j in range(4):
            inicio = j * 267 
            fim = inicio + 267
            imagem_final_unico_canal[inicio:fim, :] = image_50s[:, :, j]
        
        imagem_final_unico_canal2 = np.zeros((400, 291))
        for n in range(4):
            inicio = n * 100 
            fim = inicio + 100
            imagem_final_unico_canal2[inicio:fim, :] = image_10s[:, :, n]
        
        # Resize and combine spectrograms
        imagem_final_unico_canal_resized = cv2.resize(imagem_final_unico_canal, (400, 800), interpolation=cv2.INTER_AREA)
        imagem_final_unico_canal2_resized = cv2.resize(imagem_final_unico_canal2, (300, 400), interpolation=cv2.INTER_AREA)
        eeg_new_resized = cv2.resize(image_10m, (300, 400), interpolation=cv2.INTER_AREA)
        
        # Create final combined image
        imagem_final = np.zeros((800, 700), dtype=np.float32)
        imagem_final[0:800, 0:400] = imagem_final_unico_canal_resized
        imagem_final[0:400, 400:700] = imagem_final_unico_canal2_resized
        imagem_final[400:800, 400:700] = eeg_new_resized
        imagem_final = imagem_final[::-1]  # flip vertically
        
        # Resize to target size
        imagem_final = cv2.resize(imagem_final, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Save to dictionary
        all_eegs[eeg_new_key] = imagem_final
        
        # Show the first example
        if i == 0:
            plt.figure(figsize=(10, 10))
            plt.imshow(imagem_final, cmap='jet')
            plt.axis('off')
            plt.title(f"Combined Spectrogram for EEG {eeg_id}")
            plt.savefig(f"{output_folder}/example_spectrogram.png")
            plt.close()
    
    return all_eegs


def create_custom_spectrogram(data):
    """
    Create a custom spectrogram from EEG data using the bipolar montage approach.
    
    Args:
        data (pd.DataFrame): EEG data
        
    Returns:
        np.ndarray: Spectrogram data
    """
    # Spectrogram parameters
    nperseg = 150  # Length of each segment
    noverlap = 128  # Overlap between segments
    NFFT = max(256, 2 ** int(np.ceil(np.log2(nperseg))))

    # LL Spec = ( spec(Fp1 - F7) + spec(F7 - T3) + spec(T3 - T5) + spec(T5 - O1) )/4
    freqs, t, spectrum_LL1 = signal.spectrogram(data['Fp1']-data['F7'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL2 = signal.spectrogram(data['F7']-data['T3'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL3 = signal.spectrogram(data['T3']-data['T5'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL4 = signal.spectrogram(data['T5']-data['O1'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    LL = (spectrum_LL1 + spectrum_LL2 + spectrum_LL3 + spectrum_LL4)/4

    # LP Spec = ( spec(Fp1 - F3) + spec(F3 - C3) + spec(C3 - P3) + spec(P3 - O1) )/4
    freqs, t, spectrum_LP1 = signal.spectrogram(data['Fp1']-data['F3'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP2 = signal.spectrogram(data['F3']-data['C3'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP3 = signal.spectrogram(data['C3']-data['P3'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP4 = signal.spectrogram(data['P3']-data['O1'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    LP = (spectrum_LP1 + spectrum_LP2 + spectrum_LP3 + spectrum_LP4)/4

    # RP Spec = ( spec(Fp2 - F4) + spec(F4 - C4) + spec(C4 - P4) + spec(P4 - O2) )/4
    freqs, t, spectrum_RP1 = signal.spectrogram(data['Fp2']-data['F4'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP2 = signal.spectrogram(data['F4']-data['C4'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP3 = signal.spectrogram(data['C4']-data['P4'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP4 = signal.spectrogram(data['P4']-data['O2'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    RP = (spectrum_RP1 + spectrum_RP2 + spectrum_RP3 + spectrum_RP4)/4

    # RL Spec = ( spec(Fp2 - F8) + spec(F8 - T4) + spec(T4 - T6) + spec(T6 - O2) )/4
    freqs, t, spectrum_RL1 = signal.spectrogram(data['Fp2']-data['F8'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL2 = signal.spectrogram(data['F8']-data['T4'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL3 = signal.spectrogram(data['T4']-data['T6'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL4 = signal.spectrogram(data['T6']-data['O2'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    
    RL = (spectrum_RL1 + spectrum_RL2 + spectrum_RL3 + spectrum_RL4)/4
    
    # Concatenate all spectrograms
    spectrogram = np.concatenate((LL, LP, RP, RL), axis=0)
    
    return spectrogram


def preprocess_spectrogram(data, normalization='instance'):
    """
    Preprocess a spectrogram for model input.
    
    Args:
        data (np.ndarray): Spectrogram data
        normalization (str, optional): Normalization method. Defaults to 'instance'.
            Options: 'instance', 'datawide'
            
    Returns:
        np.ndarray: Processed spectrogram
    """
    # Handle NaN values
    mask = np.isnan(data)
    data[mask] = -1
    
    # Clip and log transform
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)
    
    # Normalize
    eps = 1e-6
    if normalization == 'instance':
        # Instance-wise normalization
        data_mean = data.mean()
        data_std = data.std()
        data = (data - data_mean) / (data_std + eps)
    elif normalization == 'datawide':
        # Dataset-wide normalization (using pre-calculated values)
        dataset_wide_mean = -0.2972692229201065
        dataset_wide_std = 2.5997336315611026
        data = (data - dataset_wide_mean) / (dataset_wide_std + eps)
    
    return data


def create_parquet_spectrogram(path_to_parquet):
    """
    Create a spectrogram from a parquet file.
    
    Args:
        path_to_parquet (str): Path to parquet file
        
    Returns:
        np.ndarray: Processed spectrogram
    """
    data = pd.read_parquet(path_to_parquet)
    data = data.fillna(-1).values[:, 1:].T
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)
    
    return data
