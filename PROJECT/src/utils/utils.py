import polars as pl
import scipy.io
from pathlib import Path

def get_file_names(folder_path="data"):
    folder = Path(folder_path)
    mat_files = list(folder.glob("*.mat"))
    
    print(f"Found {len(mat_files)} .mat files in {folder_path}:")
    
    return mat_files

 
# Read Access tables
def read_file(file_path):
    try:
        # Load MATLAB file
        mat_data = scipy.io.loadmat(file_path)
        
        # Convert to Polars DataFrame
        # You'll need to inspect the structure of your .mat file
        for key, value in mat_data.items():
            if not key.startswith('__'):  # Skip metadata
                if hasattr(value, 'shape'):  # It's likely array data
                    print(f"Found array '{key}' with shape {value.shape}")
                    # Convert to DataFrame
                    df = pl.from_numpy(value)
                    return df
        
        print("Available keys in .mat file:", [k for k in mat_data.keys() if not k.startswith('__')])
        return None
        
    except Exception as e:
        print(f"Error reading MATLAB file: {e}")
        return None

