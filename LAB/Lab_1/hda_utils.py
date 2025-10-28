import numpy as np
import os
import pandas as pd
from matplotlib.gridspec import GridSpec
from ecgdetectors import Detectors
from scipy.signal import resample, medfilt
from scipy.io import loadmat
from scipy.stats import skew, kurtosis  
from EntropyHub import ApEn, SampEn

dataset_path = os.path.join("..","Datasets")

DATA_FILE_PATH = os.path.join(dataset_path, "Lab_1", "REMOCOP_20211022_17h20.csv" )

# DECODING FUNCTIONS
def convert_array_to_signed_int(data, offset, length):
    return int.from_bytes(
        bytearray(data[offset: offset + length]), byteorder="little", signed=True,
    )


def convert_to_unsigned_long(data, offset, length):
    return int.from_bytes(
        bytearray(data[offset: offset + length]), byteorder="little", signed=False,
    )


def data_conv2(data):
    ecg_session_data = []
    ecg_session_time = []
    if len(data)>0:
        tmp = data[0]
    else:
        tmp = 0x00

    if tmp == 0x00:
        timestamp = convert_to_unsigned_long(data, 1, 8)
        step = 3
        samples = data[10:]
        offset = 0
        while offset < len(samples):
            ecg = convert_array_to_signed_int(samples, offset, step)
            offset += step
            ecg_session_data.extend([ecg])
            ecg_session_time.extend([timestamp])
    return ecg_session_data

# MAIN FUNCTION
def load_dataset():

    # Load data
    data_df = pd.read_csv(DATA_FILE_PATH, sep='\t')

    # ppg_df = data_df["PPGvalue"].dropna()
    # ppg_signal = np.array(ppg_df).reshape(-1,1)

    # This step is necessary because the file contains less ECG packets than PPG
    ecg_df = data_df["ECGString"].dropna()
    
    ecg_signal = []
    for sample in ecg_df:
        array_data = bytearray()
        vec = np.arange(0, len(sample), 2)

        for index in vec:
            tmp = sample[index:index + 2]
            tmp2 = int(tmp, 16)
            array_data.append(int(tmp2))
        ecg_track = data_conv2(array_data)
    
        ecg_signal.extend(ecg_track)
    
    ecg_signal = np.array(ecg_signal).reshape(-1,1)

    return ecg_signal


def load_dataset_DBSCAN():
    def baseline_wander_removal(data):
        fs = 300
        
        win_size = int(np.round(0.2 * fs)) + 1
        baseline = medfilt(data, win_size)        
        win_size = int(np.round(0.6 * fs)) + 1 
        baseline = medfilt(baseline, win_size)
        
        # Removing baseline
        filt_data = data - baseline
        return filt_data
    
    file_name_A = os.path.join(dataset_path, "Lab_1", "A08283.mat")
    file_name_N = os.path.join(dataset_path, "Lab_1", "A00001.mat")

    if isinstance(file_name_A, bytes):
        file_name_A = file_name_A.decode()
    ecg_signal_A = loadmat(file_name_A)['val'][0]

    if isinstance(file_name_N, bytes):
        file_name_N = file_name_N.decode()
    ecg_signal_N = loadmat(file_name_N)['val'][0]

    ecg_signal_A = baseline_wander_removal(ecg_signal_A)
    ecg_signal_A = np.expand_dims(ecg_signal_A, axis=1)

    ecg_signal_N = baseline_wander_removal(ecg_signal_N)
    ecg_signal_N = np.expand_dims(ecg_signal_N, axis=1)

    ecg_signal = np.concatenate((ecg_signal_A, ecg_signal_N), axis=0)
    return ecg_signal

# ECG segmentation
def segment_ECG(ecg_signal, fs = 130, word_len = 100):
    detectors = Detectors(fs)
    r_peaks = detectors.two_average_detector(np.squeeze(ecg_signal))
    ecg_matrix = []
    original_len = []
    for i in range(len(r_peaks)-1):
        ecg_segment = np.array((ecg_signal[r_peaks[i]:r_peaks[i+1]]).reshape(1,-1)[0])
        original_len.append(len(ecg_segment))
        ecg_word = resample(ecg_segment, 100)
        ecg_matrix.append(ecg_word)

    ecg_matrix = np.array(ecg_matrix)
    return ecg_matrix, r_peaks, original_len

def feature_extract(ecg_segment):
    ## 1) Compute length of the signal
    signal_len = len(ecg_segment)
    ## 2) Compute standard deviation of the signal
    signal_std = np.std(ecg_segment)
    ## 3) Compute the sample entropy of the signal
    signal_samp_ent, _, _ = SampEn(ecg_segment)
    ## 4) Compute the approximate entropy of the signal
    signal_app_ent, _ = ApEn(ecg_segment)
    ## 5) Compute the integral of the signal
    signal_int = np.sum(ecg_segment)
    ## 6) Compute the x location of the peak 
    signal_peak = np.argmax(ecg_segment)
    ## 7) Compute the y location of the peak
    signal_peak_value = np.max(ecg_segment)
    ## 8) Compute the x location of the second highest peak
    signal_second_peak = np.argsort(ecg_segment)[-2]
    ## 9) Compute the y location of the second highest peak
    signal_second_peak_value = np.sort(ecg_segment)[-2]
    ## 10) Compute the skewness of the signal
    signal_skew = skew(ecg_segment)
    ## 11) Compute the kurtosis of the signal
    signal_kurt = kurtosis(ecg_segment)
    return np.array([signal_len,
                     signal_std,
                     signal_samp_ent[0],
                     signal_app_ent[0],
                     signal_int,
                     signal_peak,
                     signal_peak_value,
                     signal_second_peak,
                     signal_second_peak_value,
                     signal_skew,
                     signal_kurt])


def segment_ECG2(ecg_signal, fs = 130, word_len = 100):
    detectors = Detectors(fs)
    r_peaks = detectors.two_average_detector(np.squeeze(ecg_signal))
    feature_matrix = []
    segments_list = []
    original_len = []
    for i in range(len(r_peaks)-1):
        ecg_segment = np.array((ecg_signal[r_peaks[i]:r_peaks[i+1]]).reshape(1,-1)[0])
        original_len.append(len(ecg_segment))
        
        ## Compute FFT of the signal
        w = np.fft.fft(ecg_segment)
        freqs = np.fft.fftfreq(len(ecg_segment), 1/fs)

        positive_freqs = freqs[freqs >= 0]
        positive_w = w[freqs >= 0]

        ecg_features = feature_extract(ecg_segment)
        ecg_fft_features = feature_extract(np.abs(positive_w))

        ecg_word = np.concatenate((ecg_features, ecg_fft_features), axis=0)

        feature_matrix.append(ecg_word)
        segments_list.append(ecg_segment)

    feature_matrix = np.array(feature_matrix)
    return feature_matrix, segments_list, r_peaks, original_len



# Signal reconstruction
def matrix_to_signal(matrix, original_len = None):
    if original_len == None:
        signal = matrix.reshape(-1,1,order = 'C')
    else:  
        signal = []
        for i in range(len(original_len)):
            ecg_word = resample(matrix[-len(original_len)+i,:], original_len[-len(original_len)+i])
            signal.extend(ecg_word)
        signal = np.array(signal)
    return signal

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    
    ecg_signal = load_dataset_DBSCAN()

    # ECG segmentation
    fs = 300
    ecg_mat, _, r_peaks, original_len = segment_ECG2(ecg_signal, fs)

    # Normalize data along feature axis (2 lines of code)
    scaler = StandardScaler().fit(ecg_mat)
    ecg_mat_norm = scaler.transform(ecg_mat)

    print(ecg_mat_norm.shape)
    print(ecg_mat_norm)