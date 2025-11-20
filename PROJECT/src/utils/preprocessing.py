from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import polars as pl

import numpy as np
#####################################################################

def normalize_eeg(eeg_data):
    X = eeg_data.to_numpy()

    # Normalize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Convert back to Polars, preserving column names
    return pl.DataFrame(X_norm, schema=eeg_data.columns)


####### COMPRESSION ###########


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
    
   
