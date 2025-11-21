#IMPORT
import polars as pl
import scipy.io
from pathlib import Path
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import mne


#####################################################################
def get_file_names(folder_path="data"):
    folder = Path(folder_path)
    mat_files = list(folder.glob("*.mat"))
    
    print(f"Found {len(mat_files)} .mat files in {folder_path}:")
    
    return mat_files

def read_file(file_path):
    """Returns both metadata and EEG data as separate Polars DataFrames"""
    try:
        mat_data = scipy.io.loadmat(file_path)
        o_data = mat_data['o'][0, 0]
        subject_info = ((file_path.split("/")[-1]).split(".")[0]).split("-")

      # Create metadata DataFrame
        info = {
            'id': str(o_data['id'][0]) if o_data['id'].size > 0 else "Unknown",
            'exp':  str(subject_info[0]),
            'subject': str(subject_info[1][-1]),
            'subject_sex': str("M") if str(subject_info[1][-1]) in ["A","B","C","D","F","G","H","K"] else str("F"),
            'subject_age':  "[25-30]" if str(subject_info[1][-1]) in ["C", "D"]
                            else "[30-35]" if str(subject_info[1][-1]) in ["f", "g"]
                            else "[20-25]",
            'date': str(subject_info[2]),            
            'samples': int(o_data['nS'][0, 0]) if o_data['nS'].size > 0 else 0,
            'sampling_freq': int(o_data['sampFreq'][0, 0]) if o_data['sampFreq'].size > 0 else 0,
            'channels': len(o_data['chnames']) if o_data['chnames'].size > 0 else 0
        }
        
        df_metadata = pl.DataFrame([info])
      

        # Extract channel names
        channel_names = [str(o_data["chnames"][i][0]).replace("[", "").replace("]", "").replace("'", "").strip() for i in range(o_data["chnames"].shape[0])]
   
        # Create DataFrame - use schema as simple list of names
        df_data = pl.from_numpy(o_data['data'])
        df_data = df_data.rename({f"column_{i}": name for i, name in enumerate(channel_names)})

        # Add labels/markers
        markers = o_data['marker'].flatten()  # Flatten to 1D array
        df_data = df_data.with_columns(marker=pl.Series(markers))
    
        return df_metadata, df_data
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None
    
def read_file_to_raw(file_path, drop_channels=['X5']):
    try:
        mat_data = scipy.io.loadmat(file_path)
        subject_info = ((file_path.split("/")[-1]).split(".")[0]).split("-")
        o_data = mat_data['o'][0, 0]        
        sfreq = int(o_data['sampFreq'][0, 0]) if o_data['sampFreq'].size > 0 else 200
        channel_names = []
        all_channel_names = []  # Keep track of all original channels
        for i in range(o_data["chnames"].shape[0]):
            ch_name = str(o_data["chnames"][i][0]).replace("[", "").replace("]", "").replace("'", "").strip()
            all_channel_names.append(ch_name)
            if ch_name not in drop_channels:
                channel_names.append(ch_name)
        
        print(f"Channels: {channel_names}")

        full_data = o_data['data']  
        keep_indices = [i for i, ch_name in enumerate(all_channel_names) if ch_name not in drop_channels]
        eeg_data = full_data[:, keep_indices].T  # Shape: (n_channels, n_samples)
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=sfreq,
            ch_types='eeg'
        )
        
        # Add standard 10-20 montage
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)
        raw = mne.io.RawArray(eeg_data, info)
        print("Successfully created Raw object")
        return raw
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_eeg_signals(
    eeg_data,
    first_sample=0,
    window_size=None,       
    freq=200,
    channels=None           
):
    dt = 1 / freq
    total_samples = eeg_data.height

    if channels is None:
        channels = [col for col in eeg_data.columns if col != "marker"]
    else:
        channels = [ch for ch in channels if ch in eeg_data.columns]

    if window_size is None:
        last_sample = total_samples
    else:
        samples_in_window = int(window_size)
        last_sample = min(first_sample + samples_in_window, total_samples)

    eeg_slice = eeg_data.slice(first_sample, last_sample - first_sample)

    # Time axis
    N = eeg_slice.height
    t = np.arange(N) * dt

    # Plot
    plt.figure(figsize=(12, 6))
    for ch in channels:
        y = eeg_slice[ch].to_numpy()
        plt.plot(t, y, label=ch)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

#A plot that shows: “How much of my EEG signal is present at each frequency?”
#Useful to identify dominant frequencies, artifacts, and overall spectral characteristics of the EEG data.
#THIS CAN BE DELETED SINCE WE ARE USING RAW NOW

def power_spectrum(eeg_data, fs=200):
    # Select numeric columns (all EEG channels)
    columns = [col for col in eeg_data.columns if col != "marker"]
    print("Processing channels:", columns)
    
    for col in columns:
        data = eeg_data[col].to_numpy()
        
        # Compute Welch PSD
        freqs, psd = welch(data, fs=fs, nperseg=1024)
        
        # Detect line noise frequency in 40–70 Hz
        mask = (freqs >= 40) & (freqs <= 70)
        line_freq = freqs[mask][np.argmax(psd[mask])]
        
        print(f"Channel {col} → likely line noise: {line_freq:.2f} Hz")
        
        # Plot PSD
        plt.figure(figsize=(6, 3))
        plt.semilogy(freqs, psd)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title(f"Power Spectrum of {col}")
        plt.tight_layout()
        plt.show()
