#IMPORT
import polars as pl
import scipy.io
from pathlib import Path
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import mne
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#####################################################################

def get_file_names(folder_path="../data", experiment=None):
    folder = Path(folder_path)
    mat_files = [file.name for file in folder.glob("*.mat")]
    if experiment:
        mat_files = [file for file in mat_files if file.startswith(experiment + "-")]
    mat_files = [str(folder / file).replace('\\', '/') for file in mat_files]
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

    eeg_slice_plot = eeg_slice.with_columns([
        pl.col("marker") * 10
    ]) if "marker" in channels else eeg_slice
    # Plot
    plt.figure(figsize=(12, 6))
    # Plot EEG channels
    for ch in channels:
        if ch != "marker":
            y = eeg_slice[ch].to_numpy()
            plt.plot(t, y, label=ch)
    
    # Plot marker * 10 as integer steps
    if "marker" in channels:
        marker_data = eeg_slice["marker"].to_numpy() * 10
        plt.plot(t, marker_data, label="marker ×10", color='red', linewidth=2, linestyle='--')

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


def plot_eeg_signals_with_events(
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot EEG channels on top
    for ch in channels:
        if ch != "marker":
            y = eeg_slice[ch].to_numpy()
            ax1.plot(t, y, label=ch, alpha=0.7)
    ax1.set_ylabel('EEG Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot marker events on bottom with emphasis
    if "marker" in channels:
        marker_data = eeg_slice["marker"].to_numpy()
        
        # Plot the raw marker data
        ax2.plot(t, marker_data, label='Raw Marker', alpha=0.3, color='gray')
        
        # Highlight non-zero events
        event_mask = marker_data != 0
        if np.any(event_mask):
            ax2.scatter(t[event_mask], marker_data[event_mask] * 10, 
                       color='red', s=30, label='Events ×10', zorder=5)
            
            # Add text labels for event values
            for i in np.where(event_mask)[0]:
                ax2.text(t[i], marker_data[i] * 10 + 0.5, f'{marker_data[i]}', 
                        ha='center', va='bottom', fontsize=8, color='red')
        
        ax2.set_ylabel('Marker Value')
        ax2.set_xlabel('Time (s)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1, max(35, marker_data.max() * 10 + 5))  # Adjust ylim for visibility
    
    plt.tight_layout()
    plt.show()
    
    # Print event statistics for this window
    if "marker" in channels:
        window_events = marker_data[marker_data != 0]
        if len(window_events) > 0:
            unique_events, event_counts = np.unique(window_events, return_counts=True)
            print(f"Events in this window: {dict(zip(unique_events, event_counts))}")
            

import mne
import numpy as np
from pathlib import Path

def create_epochs_from_your_data(eeg_data, marker_data, sfreq=200, epoch_duration=4, baseline_duration=0.2):
    """
    Convert your continuous EEG data to epochs for classification
    
    Parameters:
    - eeg_data: Your EEG signals (channels x time)
    - marker_data: Event markers with labels
    - sfreq: Sampling frequency (200 Hz)
    - epoch_duration: Length of each epoch in seconds
    - baseline_duration: Baseline period before events
    """
    
    # 1. Find event positions and labels from your marker channel
    events = find_events_from_marker(marker_data, sfreq)
    
    # 2. Define your channel information
    ch_names = ['X1', 'X2', 'X3', 'X4', 'X5']  # Adjust based on your channels
    ch_types = ['eeg'] * len(ch_names)
    
    info = mne.create_info(
        ch_names=ch_names, 
        sfreq=sfreq, 
        ch_types=ch_types
    )
    
    # 3. Create epochs around events
    epochs = mne.Epochs(
        raw=eeg_data,  # You might need to create a Raw object first
        events=events,
        tmin=-baseline_duration,  # Start before event
        tmax=epoch_duration,      # End after event
        baseline=(None, 0),       # Baseline correction
        preload=True
    )
    
    return epochs

def find_events_from_marker(marker_data):
    event_samples = np.where(np.diff(marker_data != 0) == True)[0] + 1
    # Get event values (your labels)
    event_values = marker_data[event_samples]
    # Create events array for MNE: [sample, 0, value]
    events = np.column_stack([
        event_samples, 
        np.zeros(len(event_samples)), 
        event_values
    ]).astype(int)
    
    return events


def epochs_to_polars(epochs):
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2]
    times = epochs.times
    channel_names = epochs.ch_names
    
    n_epochs, n_channels, n_times = X.shape
    
    # Reshape data to 2D: (n_epochs * n_times, n_channels)
    X_2d = X.transpose(0, 2, 1).reshape(-1, n_channels)
    
    # Create repeating arrays for metadata
    epoch_ids = np.repeat(np.arange(n_epochs), n_times)
    labels = np.repeat(y, n_times)
    time_points = np.tile(times, n_epochs)
    
    # Build dictionary for DataFrame
    data_dict = {
        'epoch_id': epoch_ids,
        'time': time_points, 
        'label': labels
    }
    
    # Add channel data
    for i, ch_name in enumerate(channel_names):
        data_dict[ch_name] = X_2d[:, i]
    
    df = pl.DataFrame(data_dict)
    return df

def quick_XGBOOST_test(df):
    exclude_cols = {'epoch_id', 'label', 'time'}
    electrode_cols = [col for col in df.columns if col not in exclude_cols]
    # Ensure the DataFrame is sorted by epoch_id and time for consistent reshaping
    df = df.sort(['epoch_id', 'time'])

    epoch_df = df.group_by('epoch_id').agg(
        pl.col('label').first().alias('label'),
        pl.col('time').count().alias('time_count')
    )

    # Check if all epochs have the same number of time steps (for reshaping)
    time_counts = epoch_df['time_count'].unique()
    if len(time_counts) > 1:
        raise ValueError("Epochs have varying time steps; padding or truncation needed.")


    num_epochs = len(epoch_df)
    time_steps = epoch_df['time_count'][0]
    num_electrodes = len(electrode_cols)
    y = epoch_df['label'].to_numpy()
    data = df.select(electrode_cols).to_numpy()

    # Reshape to 3D: (num_epochs, time_steps, num_electrodes)
    X_3d = data.reshape(num_epochs, time_steps, num_electrodes)
    # Flatten to 2D for XGBoost: (num_epochs, time_steps * num_electrodes)
    X = X_3d.reshape(num_epochs, -1)

    print(X.shape, y.shape)  # Should be (num_epochs, time_steps * num_electrodes) and (num_epochs,)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    num_classes = len(np.unique(y))
    if num_classes == 2:
        objective = 'binary:logistic'
    else:
        objective = 'multi:softmax'

    params = {'objective': objective}
    if num_classes > 2:
        params['num_class'] = num_classes

    # Initialize and train the XGBoost Classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        eval_metric='mlogloss' if num_classes > 2 else 'logloss',
        random_state=42,
        **params
    )

    print("Starting XGBoost training...")
    xgb_model.fit(X_train, y_train)
    print("Training complete.")

    # Make predictions on the test set
    y_pred = xgb_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")