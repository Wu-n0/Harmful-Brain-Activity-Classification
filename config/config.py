"""
Configuration file for HMS Brain Activity Classification models.
"""

class CFG:
    """
    Configuration class for ResNet1D model.
    """
    VERSION = 88

    model_name = "resnet1d_gru"

    seed = 2024
    batch_size = 32
    num_workers = 0

    fixed_kernel_size = 5
    kernels = [3, 5, 7, 9, 11]
    linear_layer_features = 304   # 1/5 Signal = 2_000

    seq_length = 50  # Seconds
    sampling_rate = 200  # Hz
    nsamples = seq_length * sampling_rate  # Number of samples
    out_samples = nsamples // 5

    freq_channels = []  # [(8.0, 12.0)]; [(0.5, 4.5)]
    filter_order = 2
    random_close_zone = 0.0  # 0.2
        
    target_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    map_features = [
        ("Fp1", "T3"),
        ("T3", "O1"),
        ("Fp1", "C3"),
        ("C3", "O1"),
        ("Fp2", "C4"),
        ("C4", "O2"),
        ("Fp2", "T4"),
        ("T4", "O2"),
        #('Fz', 'Cz'), ('Cz', 'Pz'),        
    ]

    eeg_features = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2", 
                    "F3", "P3", "F7", "T5", "F4", "P4", "F8", "T6", "EKG"]                    
    feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}
    simple_features = []  # 'Fz', 'Cz', 'Pz', 'EKG'

    n_map_features = len(map_features)
    in_channels = n_map_features + n_map_features * len(freq_channels) + len(simple_features)
    target_size = len(target_cols)


class EfficientNetConfig:
    """
    Configuration class for EfficientNet models.
    """
    batch_size = 32
    spec_size = (512, 512, 3)
    classes = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
    n_classes = len(classes)
    folds = 5
    epochs = 4
    
    # Normalization constants
    dataset_wide_mean = -0.2972692229201065
    dataset_wide_std = 2.5997336315611026
    ownspec_mean = 7.29084372799223e-05
    ownspec_std = 4.510082606216031
