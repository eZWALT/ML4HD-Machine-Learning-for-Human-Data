from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import polars as pl
import numpy as np
#####################################################################

######## PREPROCESSING ###########
#As general preprocessing pipeline for EEG data, we can consider the PREP pipeline.[ N. Bigdely-Shamlo, T. Mullen, C. Kothe, K. M. Su, and K. A. Robbins, \The prep pipeline: Standardized preprocessing for large-scale eeg analysis," Frontiers in Neuroinformatics, vol. 9, no. 16, 2015.]
#This is a generalized preprocessing for EEG data. We will use it as a base and then adapt it to our specific needs.
#PREP IS COMPRISED OF THE FOLLOWING STEPS:
#1. Line Noise Removal (Unwanted electrical interference from the power supply in the environment) 
# - However or data seems to have been already notched at 50Hz, as the paper suggested, so we can skip this step. 
# - Also seems to have been already filtered with 0Hz high-pass cutoff. Even if the paper says 1Hz, we can skip this step too.
#2. Robust Referencing 
# Phase 1: Estimate the true signal mean
# Phase 2: Find the bad channels relative to true mean and interpolate
#3. Bad Channel Interpolation
#4. Rereferencing or undo interpolation
# - This step we wont do it, we will just decide beforehand if we want to use average reference or not. And same for interpolation.
# After PREP we have to work on the spikes removal and artifact rejection, but this is not part of PREP. We will add this to the general preprocessing pipeline too.

#ESTO NO FUNCIONA BIEN HAY QUE REVISARLO
def robust_reference(df, std_z_thresh=3.0, flat_thresh=1e-6):
    """
    std_z_thresh :Channels with std above mean + z*std are marked bad
    flat_thresh : Channels with std < flat_thresh are marked bad
    """
    eeg_cols = [col for col in df.columns if col not in ["marker"]]
    data = df.select(eeg_cols).to_numpy() 
    
    # Detect bad channels
    ch_std = data.std(axis=0) #standard deviation per channel
    flat_idx = np.where(ch_std < flat_thresh)[0] #channel indices with low std
    z = (ch_std - ch_std.mean()) / ch_std.std() 
    high_idx = np.where(z > std_z_thresh)[0] #channel indices with high std

    bad_idx = np.unique(np.concatenate([flat_idx, high_idx]))
    bad_channels = [eeg_cols[i] for i in bad_idx]

    # Exclude bad channels for reference
    good_idx = [i for i in range(data.shape[1]) if i not in bad_idx]
    mean_signal = data[:, good_idx].mean(axis=1, keepdims=True)

    # Subtract mean from all channels
    data_ref = data - mean_signal

    # Convert back to Polars
    df_ref = pl.DataFrame({col: data_ref[:, i] for i, col in enumerate(eeg_cols)})
    
    df_ref = df_ref.with_columns(df["marker"])


    return df_ref, bad_channels


def PREP(eeg_data, std_z_thresh=3.0, flat_thresh=1e-6, interpolate_bad=True, average_reference=True):
    return None

def spike_removal(eeg_data):
    return eeg_data

#THIS CAN BE REMOVE WHEN PPREP IS FULLY IMPLEMENTED
def normalize_eeg(eeg_data):
    X = eeg_data.to_numpy()

    # Normalize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Convert back to Polars, preserving column names
    return pl.DataFrame(X_norm, schema=eeg_data.columns)


####### COMPRESSION ###########
#We can implement here different compression techniques for EEG data, such as PCA, LTC and Autoencoders. Then we can compare them and see which one works better for our specific case.

def LTC(x, epsilon):
    x = np.asarray(x)
    n = len(x)
    segments = []
    i0 = 0 
    low_slope = -np.inf
    high_slope = np.inf
    
    for i in range(1, n):
        low = (x[i] - x[i0] - epsilon) 
        high = (x[i] - x[i0] + epsilon) 
        
        # Update feasible slope interval
        low_slope = max(low_slope, low)
        high_slope = min(high_slope, high)
        
        if low_slope > high_slope:
            # Emit previous segment
            segments.append((i0, x[i0], i-1, x[i-1]))
            # Start new segment from last point
            i0 = i-1
            low_slope = -np.inf
            high_slope = np.inf
    
    # Emit final segment
    segments.append((i0, x[i0], n-1, x[-1]))
    
    return segments
    
   
