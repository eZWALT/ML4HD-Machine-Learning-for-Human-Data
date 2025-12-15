import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, AveragePooling1D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, MaxPool1D, Add, Concatenate, Reshape
from tensorflow.keras.models import Model
from typing import List

def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

#######################################################################################

# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   


def ShallowConvNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25    
    
    Note that this implementation has not been verified by the original 
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations. 
    """

    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(40, (1, 13), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


#################################################################


def Inception(nb_classes, Chans = 64, Samples = 256):
   
    input_main = Input(shape=(Chans,Samples)) # (None, 301, 19)
    print(input_main.shape)
    block1       = Inception_module(
                    input_tensor=input_main,
                    bottleneck_size=nb_classes*3
                    )
    block1       = Inception_module(
                    input_tensor=block1,
                    bottleneck_size=nb_classes*3
                    )
    block1       = Inception_module(
                    input_tensor=block1,
                    bottleneck_size=nb_classes*3
                    )
  
   # ---  Residual Mapping F(x0, {Wi}) ---
    residual_mapping = Conv1D(
        filters=block1.shape[-1],
        kernel_size=1,
        padding='same',
        activation='linear', 
    )(input_main) 
    block1 = Add()([residual_mapping, block1])
    
    block2       =  Inception_module(
                    input_tensor=block1,
                    bottleneck_size=nb_classes*3
                    )
    block2       =  Inception_module(
                    input_tensor=block2,
                    bottleneck_size=nb_classes*3
                    )   
    block2       =  Inception_module(
                    input_tensor=block2,
                    bottleneck_size=nb_classes*3
                    )
    
    # ---  Residual Mapping F(x0, {Wi}) ---
    residual_mapping = Conv1D(
        filters=block2.shape[-1],
        kernel_size=1,
        padding='same',
        activation='linear', 
    )(block1) 
    block2 = Add()([residual_mapping, block2])
    
    avg_pool_1d = AveragePooling1D(
    pool_size=2,  # Window size for averaging (e.g., reduce time_steps by half)
    strides=2,    # How far the window moves (usually same as pool_size for downsampling)
    padding='valid' # Or 'same'
    )(block2)
    flatten      = Flatten()(avg_pool_1d)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)



def Inception_module(input_tensor,bottleneck_size=3*19, conv_kernel_sizes = [20, 60, 100, 150, 200]):
    
    # --- 1. Bottleneck Layer (Conv1D with kernel size 1) ---
    bottleneck = Conv1D(
        filters=bottleneck_size,
        kernel_size=1,
        padding='same',
        activation='linear',
    )(input_tensor)
    
    # --- 2. Parallel Convolutional Layers  ---
    parallel_features = []
    
    for i, kernel_size in enumerate(conv_kernel_sizes):
        conv_branch = Conv1D(
            filters=bottleneck_size,
            kernel_size=kernel_size,
            padding='same', # 'same' padding ensures output length is same as input
            activation='linear',
        )(bottleneck)
        parallel_features.append(conv_branch)
        
    # --- 1.2 . Max Pooling Layer ---
    pool = MaxPool1D(
        pool_size=20,
        strides=1, 
        padding='same',
    )(input_tensor)

    # --- 2.2. 1x1 Convolution to Enlarge Depth ---
    
    pool_conv = Conv1D(
        filters=bottleneck_size,
        kernel_size=1,
        padding='same',
        activation='linear',
    )(pool)
    parallel_features.append(pool_conv)
        
    # --- 3. Concatenation ---
    concat = Concatenate(axis=-1,)(parallel_features)
    

    # --- 4. Batch Normalization ---
    bn = BatchNormalization()(concat)
    
    # --- 5. ReLU Activation ---
    output_tensor = Activation('relu' )(bn)
    
    return output_tensor