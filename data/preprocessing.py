"""
Data preprocessing utilities for HMS Brain Activity Classification.
This module contains functions for loading and preprocessing EEG data.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy.ndimage import gaussian_filter
import cupy as cp
import cusignal


def create_train_data():
    """
    Process training data to create datasets for model training.
    
    Returns:
        tuple: Two dataframes - high quality and low quality data based on the sum of votes
    """
    # Read the dataset
    df = pd.read_csv('../input/hms-harmful-brain-activity-classification/train.csv')
    
    # Define target classes
    classes = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
    
    # Create a new identifier combining multiple columns
    id_cols = ['eeg_id', 'spectrogram_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    df['new_id'] = df[id_cols].astype(str).agg('_'.join, axis=1)
    
    # Calculate the sum of votes for each class
    df['sum_votes'] = df[classes].sum(axis=1)
    
    # Group the data by the new identifier and aggregate various features
    agg_functions = {
        'eeg_id': 'first',
        'eeg_label_offset_seconds': ['min', 'max'],
        'spectrogram_label_offset_seconds': ['min', 'max'],
        'spectrogram_id': 'first',
        'patient_id': 'first',
        'expert_consensus': 'first',
        **{col: 'sum' for col in classes},
        'sum_votes': 'mean',
    }
    grouped_df = df.groupby('new_id').agg(agg_functions).reset_index()

    # Flatten the MultiIndex columns and adjust column names
    grouped_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped_df.columns]
    grouped_df.columns = grouped_df.columns.str.replace('_first', '').str.replace('_sum', '').str.replace('_mean', '')
    
    # Normalize the class columns
    y_data = grouped_df[classes].values
    y_data_normalized = y_data / y_data.sum(axis=1, keepdims=True)
    grouped_df[classes] = y_data_normalized

    # Split the dataset into high and low quality based on the sum of votes
    high_quality_df = grouped_df[grouped_df['sum_votes'] >= 10].reset_index(drop=True)
    low_quality_df = grouped_df[(grouped_df['sum_votes'] < 10) & (grouped_df['sum_votes'] >= 0)].reset_index(drop=True)

    return high_quality_df, low_quality_df


def eeg_from_parquet(parquet_path, eeg_features, seq_length=50, sampling_rate=200, display=False):
    """
    Read EEG data from parquet file and extract the middle portion.
    
    Args:
        parquet_path (str): Path to the parquet file
        eeg_features (list): List of EEG features to extract
        seq_length (int, optional): Duration in seconds. Defaults to 50.
        sampling_rate (int, optional): Sampling rate in Hz. Defaults to 200.
        display (bool, optional): Whether to display plots. Defaults to False.
        
    Returns:
        np.ndarray: Processed EEG data
    """
    import matplotlib.pyplot as plt
    
    # Calculate number of samples
    nsamples = seq_length * sampling_rate
    
    # Read the parquet file
    eeg = pd.read_parquet(parquet_path, columns=eeg_features)
    rows = len(eeg)

    # Get the middle portion
    offset = (rows - nsamples) // 2
    eeg = eeg.iloc[offset : offset + nsamples]

    if display:
        plt.figure(figsize=(10, 5))
        offset = 0

    # Create a placeholder with zeros
    data = np.zeros((nsamples, len(eeg_features)))

    for index, feature in enumerate(eeg_features):
        x = eeg[feature].values.astype("float32")

        # Handle NaN values
        mean = np.nanmean(x)
        nan_percentage = np.isnan(x).mean()

        if nan_percentage < 1:  # if some values are NaN, but not all
            x = np.nan_to_num(x, nan=mean)
        else:  # if all values are NaN
            x[:] = 0
        data[:, index] = x

        if display:
            if index != 0:
                offset += x.max()
            plt.plot(range(nsamples), x - offset, label=feature)
            offset -= x.min()

    if display:
        plt.legend()
        name = parquet_path.split("/")[-1].split(".")[0]
        plt.yticks([])
        plt.title(f"EEG {name}", size=16)
        plt.show()
        
    return data


def create_spectrogram_with_cusignal(eeg_data, eeg_id, start, duration=50,
                                    low_cut_freq=0.7, high_cut_freq=20, order_band=5,
                                    spec_size_freq=267, spec_size_time=30,
                                    nperseg_=1500, noverlap_=1483, nfft_=2750,
                                    sigma_gaussian=0.7, 
                                    mean_montage_names=4):
    """
    Create a spectrogram from EEG data using cusignal.
    
    Args:
        eeg_data (pd.DataFrame): EEG data
        eeg_id (int): EEG ID
        start (int): Start time in seconds
        duration (int, optional): Duration in seconds. Defaults to 50.
        low_cut_freq (float, optional): Low cut frequency for bandpass filter. Defaults to 0.7.
        high_cut_freq (float, optional): High cut frequency for bandpass filter. Defaults to 20.
        order_band (int, optional): Order of bandpass filter. Defaults to 5.
        spec_size_freq (int, optional): Frequency size of spectrogram. Defaults to 267.
        spec_size_time (int, optional): Time size of spectrogram. Defaults to 30.
        nperseg_ (int, optional): Length of each segment for spectrogram. Defaults to 1500.
        noverlap_ (int, optional): Overlap between segments. Defaults to 1483.
        nfft_ (int, optional): FFT size. Defaults to 2750.
        sigma_gaussian (float, optional): Sigma for Gaussian filter. Defaults to 0.7.
        mean_montage_names (int, optional): Number of montages to average. Defaults to 4.
        
    Returns:
        tuple: Spectrogram numpy array and processed EEG dictionary
    """
    electrode_names = ['LL', 'RL', 'LP', 'RP']

    electrode_pairs = [
        ['Fp1', 'F7', 'T3', 'T5', 'O1'],
        ['Fp2', 'F8', 'T4', 'T6', 'O2'],
        ['Fp1', 'F3', 'C3', 'P3', 'O1'],
        ['Fp2', 'F4', 'C4', 'P4', 'O2']
    ]
    
    # Filter specifications
    nyquist_freq = 0.5 * 200
    low_cut_freq_normalized = low_cut_freq / nyquist_freq
    high_cut_freq_normalized = high_cut_freq / nyquist_freq

    # Bandpass and notch filter
    bandpass_coefficients = butter(order_band, [low_cut_freq_normalized, high_cut_freq_normalized], btype='band')
    notch_coefficients = iirnotch(w0=60, Q=30, fs=200)
    
    spec_size = duration * 200
    start = start * 200
    real_start = start + (10_000//2) - (spec_size//2)
    eeg_data = eeg_data.iloc[real_start:real_start+spec_size]
    
    # Spectrogram parameters
    fs = 200
    nperseg = nperseg_
    noverlap = noverlap_
    nfft = nfft_
    
    if spec_size_freq <= 0 or spec_size_time <= 0:
        frequencias_size = int((nfft // 2)/5.15198)+1
        segmentos = int((spec_size - noverlap) / (nperseg - noverlap)) 
    else:
        frequencias_size = spec_size_freq
        segmentos = spec_size_time
        
    spectrogram = cp.zeros((frequencias_size, segmentos, 4), dtype='float32')
    
    processed_eeg = {}

    for i, name in enumerate(electrode_names):
        cols = electrode_pairs[i]
        processed_eeg[name] = np.zeros(spec_size)
        for j in range(4):
            # Compute differential signals
            signal = cp.array(eeg_data[cols[j]].values - eeg_data[cols[j+1]].values)

            # Handle NaNs
            mean_signal = cp.nanmean(signal)
            signal = cp.nan_to_num(signal, nan=mean_signal) if cp.isnan(signal).mean() < 1 else cp.zeros_like(signal)
            
            # Filter bandpass and notch
            signal_filtered = filtfilt(*notch_coefficients, signal.get())
            signal_filtered = filtfilt(*bandpass_coefficients, signal_filtered)
            signal = cp.asarray(signal_filtered)
            
            frequencies, times, Sxx = cusignal.spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

            # Filter frequency range
            valid_freqs = (frequencies >= 0.59) & (frequencies <= 20)
            frequencies_filtered = frequencies[valid_freqs]
            Sxx_filtered = Sxx[valid_freqs, :]

            # Logarithmic transformation and normalization using Cupy
            spectrogram_slice = cp.clip(Sxx_filtered, cp.exp(-4), cp.exp(6))
            spectrogram_slice = cp.log10(spectrogram_slice)

            normalization_epsilon = 1e-6
            mean = spectrogram_slice.mean(axis=(0, 1), keepdims=True)
            std = spectrogram_slice.std(axis=(0, 1), keepdims=True)
            spectrogram_slice = (spectrogram_slice - mean) / (std + normalization_epsilon)
            
            spectrogram[:, :, i] += spectrogram_slice
            processed_eeg[f'{cols[j]}_{cols[j+1]}'] = signal.get()
            processed_eeg[name] += signal.get()
        
        # AVERAGE THE MONTAGE DIFFERENCES
        if mean_montage_names > 0:
            spectrogram[:,:,i] /= mean_montage_names

    # Convert to NumPy and apply Gaussian filter
    spectrogram_np = cp.asnumpy(spectrogram)
    if sigma_gaussian > 0.0:
        spectrogram_np = gaussian_filter(spectrogram_np, sigma=sigma_gaussian)

    # Filter EKG signal
    ekg_signal_filtered = filtfilt(*notch_coefficients, eeg_data["EKG"].values)
    ekg_signal_filtered = filtfilt(*bandpass_coefficients, ekg_signal_filtered)
    processed_eeg['EKG'] = np.array(ekg_signal_filtered)

    return spectrogram_np, processed_eeg


def create_spectrogram_competition(spec_id, seconds_min):
    """
    Create a spectrogram for test data in the competition format.
    
    Args:
        spec_id (int): Spectrogram ID
        seconds_min (int): Minimum seconds
        
    Returns:
        np.ndarray: Processed spectrogram
    """
    # Read parquet file
    spec = pd.read_parquet(f'../input/hms-harmful-brain-activity-classification/test_spectrograms/{spec_id}.parquet')
    inicio = (seconds_min) // 2
    img = spec.fillna(0).values[:, 1:].T.astype("float32")
    img = img[:, inicio:inicio+300]
    
    # Log transform and normalize
    img = np.clip(img, np.exp(-4), np.exp(6))
    img = np.log(img)
    eps = 1e-6
    img_mean = img.mean()
    img_std = img.std()
    img = (img - img_mean) / (img_std + eps)
    
    return img


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a butterworth bandpass filter.
    
    Args:
        lowcut (float): Low cutoff frequency
        highcut (float): High cutoff frequency
        fs (float): Sampling frequency
        order (int, optional): Filter order. Defaults to 5.
        
    Returns:
        tuple: Filter coefficients (b, a)
    """
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a butterworth bandpass filter to data.
    
    Args:
        data (np.ndarray): Data to filter
        lowcut (float): Low cutoff frequency
        highcut (float): High cutoff frequency
        fs (float): Sampling frequency
        order (int, optional): Filter order. Defaults to 5.
        
    Returns:
        np.ndarray: Filtered data
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    """
    Apply a butterworth lowpass filter to data.
    
    Args:
        data (np.ndarray): Data to filter
        cutoff_freq (int, optional): Cutoff frequency. Defaults to 20.
        sampling_rate (int, optional): Sampling rate. Defaults to 200.
        order (int, optional): Filter order. Defaults to 4.
        
    Returns:
        np.ndarray: Filtered data
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data
