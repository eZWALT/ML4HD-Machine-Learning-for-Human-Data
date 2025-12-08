from sklearn.decomposition import PCA
import polars as pl
import numpy as np
import mne
from scipy.signal import butter, filtfilt
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

def robust_reference(epochs, std_z_thresh=3.0, flat_thresh=1e-6):
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    data = epochs.get_data()[:, picks, :]
    ch_names = [epochs.ch_names[i] for i in picks]

    ch_std = data.reshape(data.shape[0] * data.shape[2], data.shape[1]).std(axis=0)
    flat_idx = np.where(ch_std < flat_thresh)[0]
    z = (ch_std - ch_std.mean()) / ch_std.std()
    high_idx = np.where(z > std_z_thresh)[0]
    bad_idx = np.unique(np.concatenate([flat_idx, high_idx]))
    bad_channels = [ch_names[i] for i in bad_idx]
    epochs.info['bads'].extend(bad_channels)

    good_picks_idx = [i for i, name in enumerate(ch_names) if name not in bad_channels]
    good_data = data[:, good_picks_idx, :] 
    avg_ref = good_data.mean(axis=1, keepdims=True)
    data = data - avg_ref

    all_data = epochs.get_data() 
    all_data[:, picks, :] = data

    epochs._data = all_data

    return bad_channels

def PREP(epochs, high_cutoff= True, interpolate_bad=True, robust=True):
    
    if high_cutoff:
        epochs.filter(
        1.,           # l_freq: Lower cutoff frequency (1Hz) - removes slower
        40.,          # h_freq: Higher cutoff frequency (40Hz) - removes faster
        picks='eeg',  # Only apply to EEG channels (not stim channels)
        method='fir', # Finite Impulse Response filter (stable, linear phase)
        phase='zero-double' # Zero-phase filtering (no time shift in signal)
        )
    else:
        epochs.filter(1., None, picks='eeg', method='fir', phase='zero-double') 
    
    if robust:
        bad_channels=robust_reference(epochs)
        if interpolate_bad and bad_channels:
            print(f"Interpolating bad channels: {bad_channels}")
            epochs.interpolate_bads(reset_bads=True)
    
    
def remove_artifacts(epochs, spike_threshold=40e-6, expand_samples=2):
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    data = epochs.get_data()[:, picks, :] 
    
    n_epochs, n_channels, n_times = data.shape
    x = np.arange(n_times)

    # 1. Identify Spikes
    spike_idx = np.any(np.abs(data) > spike_threshold, axis=1) 
    
    if expand_samples > 0:
        expanded_idx = spike_idx.copy()
        for ep in range(n_epochs):
            epoch_spike_idx = spike_idx[ep, :]
            for shift in range(-expand_samples, expand_samples + 1):
                if shift == 0:
                    continue
                shifted = np.roll(epoch_spike_idx, shift)
                # avoid wrap-around at edges
                if shift < 0:
                    shifted[shift:] = False
                else:
                    shifted[:shift] = False
                expanded_idx[ep, :] |= shifted
        spike_idx = expanded_idx

    # 2. Mark spikes as NaN
    full_mask = np.repeat(spike_idx[:, None, :], n_channels, axis=1)
    data[full_mask] = np.nan

    # 3. Interpolate over NaNs for each EEG channel within each epoch
    for ep in range(n_epochs):
        for i in range(n_channels):
            nans = np.isnan(data[ep, i]) 
            
            if np.any(nans):
                good = ~nans
                if np.sum(good) == 0:
                    continue 
                data[ep, i, nans] = np.interp(x[nans], x[good], data[ep, i, good])

    # 4. Update Epochs object data (assign only the EEG channels back)
    epochs._data[:, picks, :] = data


def preprocess_eeg(epochs, high_cutoff= True, interpolate_bad=True, robust=True):
    PREP(epochs, high_cutoff= high_cutoff, interpolate_bad= interpolate_bad, robust=robust) 
    epochs.set_eeg_reference(ref_channels=['A1','A2'])
    epochs.drop_channels(['A1','A2'])
    #remove_artifacts(epochs)
    epochs.resample(128, npad="auto") #compression
    return epochs

 
####### DATA AUGMENTATION ###########

def extract_noise_highband(signal, fs, cutoff=100, order=8):
    nyq = fs / 2
    wn = cutoff / nyq
    b, a = butter(order, wn, btype='highpass')
    return filtfilt(b, a, signal, axis=-1)

def augment_trial(x_i, x_k, fs):
    Sn_i = extract_noise_highband(x_i, fs)
    Sn_k = extract_noise_highband(x_k, fs)
    return x_i - Sn_i + Sn_k

import numpy as np
from mne import EpochsArray

def augment_epochs(epochs, fs, ratio=2.0):
    X = epochs.get_data().copy()     
    n = len(X)
    aug_list = []

    for idx in range(n):
        for j in range(int(ratio)):
            k = np.random.choice([t for t in range(n) if t != idx])
            aug = augment_trial(X[idx], X[k], fs)
            aug_list.append(aug)

    X_aug = np.stack(aug_list)        
    X_final = np.concatenate([X, X_aug], axis=0)

    events_orig = epochs.events        
    n_aug = X_aug.shape[0]

    labels_aug = []
    for idx in range(n):
        labels_aug.extend([events_orig[idx, 2]] * int(ratio))
    labels_aug = np.array(labels_aug)

    samples_aug = np.arange(events_orig[-1, 0] + 1,
                            events_orig[-1, 0] + 1 + n_aug)

    # Create augmented events array
    events_augmented = np.zeros((n_aug, 3), dtype=int)
    events_augmented[:, 0] = samples_aug       # sample indices
    events_augmented[:, 1] = 0                 # placeholder trigger channel
    events_augmented[:, 2] = labels_aug       # keep original labels

    events_new = np.vstack([events_orig, events_augmented])

    epochs_aug = EpochsArray(
        data=X_final,
        info=epochs.info,
        events=events_new,
        tmin=epochs.tmin
    )

    return epochs_aug

    X = epochs.get_data().copy()       
    n= len(X)
    aug_list = []
    for idx in range(n):
        for j in range(int(ratio)):
            k = np.random.choice([t for t in range(n) if t != idx])
            aug = augment_trial(X[idx], X[k], fs)
            aug_list.append(aug)

    X_aug = np.stack(aug_list)
    
    # Combine with original data
    X_final = np.concatenate([X, X_aug], axis=0)
   
    events_orig = epochs.events
    n_aug = X_aug.shape[0]

    # Randomly pick original events for labels
    rand_idx = np.random.randint(0, n, size=n_aug)
    events_augmented = events_orig[rand_idx].copy()

    # Update sample numbers for augmented trials (space them after last original trial)
    events_augmented[:, 0] = np.arange(events_orig[-1, 0] + 1,
                                       events_orig[-1, 0] + 1 + n_aug)

    # Combine original + augmented events
    events_new = np.vstack([events_orig, events_augmented])

    # --- Return new EpochsArray ---
    epochs_aug = EpochsArray(
        data=X_final,
        info=epochs.info,
        events=events_new,
        tmin=epochs.tmin
    )

    return epochs_aug

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
    
   
