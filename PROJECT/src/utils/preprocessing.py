from sklearn.decomposition import PCA
import polars as pl
import numpy as np
import mne
#####################################################################

######## PREPROCESSING ###########
#As general preprocessing pipeline for EEG data, we can consider the PREP pipeline.[ N. Bigdely-Shamlo, T. Mullen, C. Kothe, K. M. Su, and K. A. Robbins, \The prep pipeline: Standardized preprocessing for large-scale eeg analysis," Frontiers in Neuroinformatics, vol. 9, no. 16, 2015.]
#This is a generalized preprocessing for EEG data. We will use it as a base and then adapt it to our specific needs.
#PREP IS COMPRISED OF THE FOLLOWING STEPS:
#1. Line Noise Removal (Unwanted electrical interference from the power supply in the environment) 
# - However or data seems to have been already notched at 50Hz, as the paper suggested, so we can skip this step.  (From the paper: "These are the hardware filters and therefore part of all the published records. Additionally, a 50 Hz notch filter is present in the EEG-1200 hardware to reduce electrical grid interference.")
# - Also seems to have been already filtered with 0Hz high-pass cutoff. Even if the paper says 1Hz, we can skip this step too. (This may not be true, so we can do it)
#2. Robust Referencing 
# Phase 1: Estimate the true signal mean
# Phase 2: Find the bad channels relative to true mean and interpolate
#3. Bad Channel Interpolation
#4. Rereferencing or undo interpolation
# - This step we wont do it, we will just decide beforehand if we want to use average reference or not. And same for interpolation.
# After PREP we have to work on the spikes removal and artifact rejection, but this is not part of PREP. We will add this to the general preprocessing pipeline too.
# For this we will use independent component analysis (ICA) to identify and remove artifacts from the EEG data.

def robust_reference(raw, std_z_thresh=3.0, flat_thresh=1e-6):
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[i] for i in picks]
    ch_std = data.std(axis=1)
    flat_idx = np.where(ch_std < flat_thresh)[0]
    z = (ch_std - ch_std.mean()) / ch_std.std()
    high_idx = np.where(z > std_z_thresh)[0]
    bad_idx = np.unique(np.concatenate([flat_idx, high_idx]))
    bad_channels = [ch_names[i] for i in bad_idx]
    raw.info['bads'] = bad_channels
    good_picks = [p for p in picks if raw.ch_names[p] not in bad_channels]
    good_data = raw.get_data(picks=good_picks)
    avg_ref = good_data.mean(axis=0)
    all_data = raw.get_data()
    all_data[picks] = all_data[picks] - avg_ref
    raw._data = all_data

    return bad_channels

def PREP(raw, high_cutoff= True, interpolate_bad=True, robust=True):
    
    if high_cutoff:
        raw.filter(
        1.,           # l_freq: Lower cutoff frequency (1Hz) - removes slower
        40.,          # h_freq: Higher cutoff frequency (40Hz) - removes faster
        picks='eeg',  # Only apply to EEG channels (not stim channels)
        method='fir', # Finite Impulse Response filter (stable, linear phase)
        phase='zero-double' # Zero-phase filtering (no time shift in signal)
        )
    else:
        raw.filter(1., None, picks='eeg', method='fir', phase='zero-double') #WE CAN PLAY WITH THE FILTERING PARAMETERS HERE I HAVE NOT TESTED THIS YET
    
    if robust:
        bad_channels=robust_reference(raw)
        if interpolate_bad and bad_channels:
            print(f"Interpolating bad channels: {bad_channels}")
            raw.interpolate_bads(reset_bads=True)
    
    raw.set_eeg_reference('average') 

def remove_artifacts(raw, spike_threshold=40e-6, expand_samples=2):
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)
    spike_idx = np.any(np.abs(data) > spike_threshold, axis=0)
    if expand_samples > 0:
        expanded_idx = spike_idx.copy()
        for shift in range(-expand_samples, expand_samples + 1):
            if shift == 0:
                continue
            shifted = np.roll(spike_idx, shift)
            # avoid wrap-around at edges
            if shift < 0:
                shifted[shift:] = False
            else:
                shifted[:shift] = False
            expanded_idx |= shifted
        spike_idx = expanded_idx

    # Mark spikes as NaN
    data[:, spike_idx] = np.nan

    # Interpolate over NaNs for each EEG channel
    n_channels = len(picks)
    n_times = data.shape[1]
    x = np.arange(n_times)
    for i in range(n_channels):
        nans = np.isnan(data[i])
        if np.any(nans):
            good = ~nans
            data[i, nans] = np.interp(x[nans], x[good], data[i, good])

    # Update raw object
    raw._data[picks, :] = data


#THIS SCALING HAS TO BE REVISED - This is just a fast implementation
def scale(raw, scale_factor=1e6):
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data, times = raw[picks, :]
    data_scaled = data / scale_factor
    raw._data[picks, :] = data_scaled

def preprocess_eeg(raw, scaling=1e6, high_cutoff= True, interpolate_bad=True, robust=True):
    scale(raw, scale_factor=scaling)
    PREP(raw, high_cutoff=  high_cutoff, interpolate_bad= interpolate_bad, robust=robust)
    remove_artifacts(raw)


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
    
   
