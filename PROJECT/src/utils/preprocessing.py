from sklearn.preprocessing import StandardScaler
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
#ESTO NO FUNCIONA BIEN HAY QUE REVISARLO
def robust_reference(raw, std_z_thresh=3.0, flat_thresh=1e-6):
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data, _ = raw[picks, :]
    ch_names = [raw.ch_names[i] for i in picks]
    
    # Detect bad channels
    ch_std = data.std(axis=1)
    flat_idx = np.where(ch_std < flat_thresh)[0]
    z = (ch_std - ch_std.mean()) / ch_std.std()
    high_idx = np.where(z > std_z_thresh)[0]
    
    bad_idx = np.unique(np.concatenate([flat_idx, high_idx]))
    bad_channels = [ch_names[i] for i in bad_idx]
    
    # Mark bad channels in the raw object
    raw.info['bads'] = bad_channels
    raw.set_eeg_reference('average')
        
    return raw, bad_channels

def PREP(raw, high_cutoff= False, interpolate_bad=True, robust=True):
    
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
        raw, bad_channels=robust_reference(raw)
        if interpolate_bad and bad_channels:
            raw.interpolate_bads(reset_bads=True)
    else:
        raw.set_eeg_reference('average') 
    return raw

def ICA(eeg_data, sfreq=200):
    # Get channel names (excluding marker if present)
    ch_names = [col for col in eeg_data.columns if col != "marker"]
    
    # Extract data without the marker column and transpose
    if "marker" in eeg_data.columns:
        data = eeg_data[ch_names].to_numpy().T  # Shape: (n_channels, n_samples)
    else:
        data = eeg_data.to_numpy().T  # Shape: (n_channels, n_samples)
    
    print(f"Data shape: {data.shape}")
    print(f"Number of channels: {len(ch_names)}")
    
    # Create MNE info structure
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )
    
    # Add standard electrode positions (important for proper visualization)
    try:
        # Try to set standard 10-20 system positions
        montage = mne.channels.make_standard_montage('standard_1020')
        # Only use the channels that exist in our data
        available_chs = [ch for ch in ch_names if ch in montage.ch_names]
        if available_chs:
            info.set_montage(montage)
            print(f"Set montage for channels: {available_chs}")
        else:
            print("Warning: No standard channel names found for montage")
    except Exception as e:
        print(f"Could not set standard montage: {e}")
    
    # Create RawArray object
    raw = mne.io.RawArray(data, info)
    
    # Apply proper band-pass filter (1-40 Hz recommended for ICA)
    print("Applying band-pass filter (1-40 Hz)...")
    raw.filter(1., 40., picks='eeg', method='fir', phase='zero-double')
    
    # Perform ICA
    print("Fitting ICA (this may take a while)...")
    ica = mne.preprocessing.ICA(
        n_components=min(20, data.shape[0]),
        random_state=42, 
        method='fastica',
        fit_params=dict(tol=1e-4)  # Convergence tolerance
    )
    ica.fit(raw)
    
    # Plot components to identify artifacts
    print("Plotting ICA components...")
    ica.plot_components()
    
    # Also plot the sources to help identify artifacts
    print("Plotting ICA sources...")
    ica.plot_sources(raw)
    
    return ica, raw

#THIS SCALING HAS TO BE REVISED - This is just a fast implementation
def scale(raw, scale_factor=1e6):
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data, times = raw[picks, :]
    data_scaled = data / scale_factor
    raw._data[picks, :] = data_scaled
    
    return raw


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
    
   
