import tensorflow as tf
from tensorflow.keras import layers
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import skimage.transform
from cnn_utils import *
from dataset_utils import *
from sklearn.model_selection import train_test_split
import keras_utils
import random
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, ELU, AveragePooling2D, Dropout, Softmax, Flatten
from tensorflow.keras.constraints import max_norm


np.random.seed(4)
tf.keras.utils.set_random_seed(21)

#PAPER BASELINE LOOK AT TABLE 2 OF V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces, Journal of neural engineering, vol. 15, no. 5, p. 056013, 2018.
def convolutional_model(
    C,          # number of EEG channels
    T,          # number of time samples
    F1=8,       # number of temporal filters in block 1
    D=2,        # number of spatial filters per temporal filter
    F2=16,      # number of pointwise filters in block 2
    N=4,        # number of output classes
    dropout_rate=0.5
):
    # --------------------
    # Block 1
    # --------------------
    input_eeg = Input(shape=(1, C, T))  # Input + Reshape
    
    # Temporal convolution: (1, 64) kernel
    x = Conv2D(F1, (1, 64), padding='same')(input_eeg)

    x = BatchNormalization(axis=-1)(x)

    # Spatial convolution: Depthwise (C, 1) per temporal filter
    x = DepthwiseConv2D((C, 1), depth_multiplier=D, depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization(axis=-1)(x)
    x = ELU()(x)
    
    # Average pooling (1,4) to reduce sampling rate
    x = AveragePooling2D(pool_size=(1, 4))(x)
    # Dropout
    x = Dropout(dropout_rate)(x)


    # --------------------
    # Block 2
    # --------------------
    x = SeparableConv2D(F2, (1, 16), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = ELU()(x)
    # Average pooling (1,8) 
    x = AveragePooling2D(pool_size=(1, 8))(x)
    x = Dropout(dropout_rate)(x)



    # --------------------
    # Classification
    # --------------------
    x = Flatten()(x)
    output = tf.keras.layers.Dense(N, activation='softmax')(x)

    model = Model(inputs=input_eeg, outputs=output)
    return model
