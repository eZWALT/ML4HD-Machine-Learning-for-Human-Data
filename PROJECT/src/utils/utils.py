import polars as pl
import scipy.io
from pathlib import Path
import numpy as np

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
        channel_names = [str(o_data["chnames"][i][0]) for i in range(o_data["chnames"].shape[0])]

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

