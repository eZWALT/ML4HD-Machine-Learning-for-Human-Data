from typing import List
import tensorflow as tf
from tensorflow.keras.layers import Layer 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  LSTM, GRU
from tensorflow.keras.layers import Conv1D, MaxPool1D, Add, Concatenate, Reshape, Permute
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D, SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Add, MultiHeadAttention
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def Simple_CNN(nb_classes, input_shape=(36, 3, 1)):
    input1 = Input(shape=input_shape)
    
    # Use larger kernels since the "image" is tiny
    x = Conv2D(32, (3, 3), activation='elu', padding='same')(input1)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (3, 3), activation='elu', padding='same')(x)
    x = BatchNormalization()(x)

    x = Flatten()(x) # Flatten is better than GlobalAverage for tiny inputs
    x = Dense(128, activation='elu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(nb_classes, activation='softmax')(x)
    return Model(inputs=input1, outputs=outputs)


def Simple_LSTM(nb_classes, input_shape=(3, 187)):
    input1   = Input(shape =input_shape)
    # 1. Normalize the CSP inputs immediately
    x = BatchNormalization()(input1)
    
    # 2. Stacked LSTM with L2 Regularization
    # We use return_sequences=True to pass the sequence to the next LSTM
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    x = LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    
    x = LSTM(32, return_sequences=False, kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    # 3. Dense classification head
    x = Dense(64, activation='elu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    
    dense = Dense(nb_classes)(x)
    softmax = Activation('softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


def Simple_GRU(nb_classes, input_shape=(3, 187)):
    input1   = Input(shape =input_shape)
    # input_shape: (Time_Steps, Features) -> (3, 187)
    x = BatchNormalization()(input1)
    
    x = GRU(128, kernel_regularizer=l2(0.01), return_sequences=False)(x)
    x = BatchNormalization()(x)
    
    # Deep Dense block to process the GRU output
    x = Dense(64, activation='elu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    
    dense = Dense(nb_classes)(x)
    softmax = Activation('softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

##################################3
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


####################
# Conformer Models #
####################

############ COMMON MODULES

# CNN Stem -> Tokens
# ShallowConvNet style of embeddings
def PatchEmbedding(
    input_tensor,
    n_filters_time=40,
    filter_time_length=25,
    n_chans=64,
    pool_time_length=75,
    pool_time_stride=15,
    drop_prob=0.5
):
    # Temporal Convolution (1 x Lt) . Frequency like
    x = Conv2D(
        n_filters_time, 
        (1, filter_time_length),
        padding="valid",
        use_bias=True
    )(input_tensor)
    
    # Spatial Convolution (N Channels x 1). Sensor Mixing
    x = Conv2D(
        n_filters_time, 
        (n_chans, 1),
        padding="valid",
        use_bias=True,
    )(x)

    # Batch Normalization, activation and temporal pooling --> patches and dropout
    # Pooling defines the token size!
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D(
        pool_size=(1, pool_time_length),
        strides=(1, pool_time_stride),
    )(x)
    x = Dropout(drop_prob)(x)
    # 1x1 projection
    x = Conv2D(n_filters_time, (1, 1), padding="same")(x)
    # (B, D, 1, T) → (B, T, D)
    x = Reshape((-1, n_filters_time))(x)
    return x

# EEGNet style of embeddings
def CTNetPatchEmbedding(
    x,
    n_chans,
    emb_size=40,
    kernel_size=64,
    depth_multiplier=2,
    pool_size_1=8,
    pool_size_2=8,
    drop_rate=0.3,
):
    # Temporal conv 
    x = Conv2D(
        emb_size // 2,
        (1, kernel_size),
        padding="same",
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)

    # Spatial filtering (Depthwise)
    x = Conv2D(
        (emb_size // 2) * depth_multiplier,
        (n_chans, 1),
        groups=emb_size // 2,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = AveragePooling2D((1, pool_size_1))(x)
    x = Dropout(drop_rate)(x)

    # Refinement
    x = Conv2D(
        emb_size,
        (1, 16),
        padding="same",
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = AveragePooling2D((1, pool_size_2))(x)
    x = Dropout(drop_rate)(x)

    # tokens
    x = Reshape((-1, emb_size))(x)
    return x


def TransformerEncoderBlock(
    x, 
    emb_size,
    num_heads,
    att_drop=0.5,
    ff_expansion=4,
): 
    # --- SELF ATTENTION ---
    x_norm = LayerNormalization(epsilon=1e-6)(x)
    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=emb_size // num_heads,
        dropout=att_drop,
    )(x_norm, x_norm)
    # Dropout + skip connection
    attn = Dropout(att_drop)(attn)
    x = Add()([x, attn])

    # --- FEED FROWARD ---
    x_norm = LayerNormalization(epsilon=1e-6)(x)
    ff = Dense(ff_expansion * emb_size, activation="gelu")(x_norm)
    # 2 dense layers, dropout + skip connection
    ff = Dropout(att_drop)(ff)
    ff = Dense(emb_size)(ff)
    ff = Dropout(att_drop)(ff)
    x = Add()([x, ff]) 
    
    return x

def TransformerEncoder(
        x, 
        num_layers,
        emb_size,
        num_heads,
        att_drop,
):
    for _ in range(num_layers):
        x = TransformerEncoderBlock(
            x, 
            emb_size=emb_size,
            num_heads=num_heads,
            att_drop=att_drop,
        )
    return x 

def ClassificationHead(
    x,
    nb_classes,
    hidden_units=(256, 32),
    drop_probs=(0.5, 0.3),
    activation="elu",
    use_batchnorm=False,
):
    """
    Generic classification head (N Layers).
    ----------
    x : tf.Tensor
        Input feature tensor.
    nb_classes : int
        Number of output classes.
    hidden_units : tuple[int]
        Sizes of hidden dense layers.
    activation : str
        Activation function for hidden layers.
    drop_probs : tuple[float]
        Dropout probability after each hidden layer.
    use_batchnorm : bool
        Whether to apply BatchNorm after dense layers.
    """
    x = Flatten()(x)

    for i, units in enumerate(hidden_units):
        x = Dense(units, activation=activation)(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        if i < len(drop_probs):
            x = Dropout(drop_probs[i])(x)

    return Dense(nb_classes, activation="softmax")(x)


def PositionalEncoding(x, seq_len, emb_size, drop_rate=0.1):
    pos_emb = tf.Variable(
        initial_value=tf.random.normal((1, seq_len, emb_size)),
        trainable=True,
        name="pos_embedding",
    )
    x = x + pos_emb[:, :tf.shape(x)[1], :]
    return Dropout(drop_rate)(x)
    

############ COMMON MODULES

# From the original paper: The results show that our model is insensitive to the depth and head number of the self-attention module while processing EEG data
# From the original paper: We design a novel visualization based on class activation mapping and topography to illustrate how the model learns essential features from a global perspective.
# preprocessing of paper:  
"""
s. Without introducing additional task-dependent prior
knowledge, we only use a few steps to pre-process the raw
EEG data. First, band-pass filtering is employed to filter out
extraneous high and low-frequency noise. Here, we use a
6-order Chebyshev filter to preserve task-relevant rhythms.
Then, a Z-score standardization is performed to reduce the
fluctuation and nonstationarity
"""
def EEGConformer(
    nb_classes,
    Chans = 64,
    Samples = 256,
    n_filters_time=40, 
    filter_time_length=25,
    pool_time_length=75,
    pool_time_stride=15,
    drop_prob=0.5,
    num_layers=6,
    num_heads=10,
    att_drop_prob=0.5
) -> Model:

    input_eeg = Input(shape=(Chans, Samples, 1))

    # 1. Patch embedding. Use a CNN to get embeddings from temporal windows
    # EEG signal (64 x 256) -> (CNN) -> 40 Filters -> (Pooling) -> T patches of 40-D vectors 
    x = PatchEmbedding(
        input_tensor=input_eeg,
        n_filters_time=n_filters_time, 
        filter_time_length=filter_time_length,
        n_chans=Chans,
        pool_time_length=pool_time_length, 
        pool_time_stride=pool_time_stride,
        drop_prob=drop_prob
    )

    # 2. Transformer Encoder
    x = TransformerEncoder(
        x, 
        num_layers=num_layers,
        emb_size=n_filters_time,
        num_heads=num_heads,
        att_drop=att_drop_prob,
    )

    # 3. Final classification head 
    x = ClassificationHead(x, nb_classes, use_batchnorm=False)
    return Model(inputs=input_eeg, outputs=x)

def CTNet(
    nb_classes,
    Chans=64,
    Samples=256,
    emb_size=40,
    num_heads=4,
    num_layers=6,
    cnn_drop_rate=0.3,
    attn_drop_rate=0.1,
    final_drop_rate=0.5,
) -> Model:

    inputs = Input(shape=(Chans, Samples))

    # (B, 1, C, T)
    x = Reshape((1, Chans, Samples))(inputs)

    # 1. CNN patch embedding (EEGNet style)
    cnn_tokens = CTNetPatchEmbedding(
        x,
        n_chans=Chans,
        emb_size=emb_size,
        drop_rate=cnn_drop_rate,
    )

    # Scale embeddings (standard transformer trick apparently)
    cnn_tokens = cnn_tokens * tf.math.sqrt(tf.cast(emb_size, tf.float32))

    # 2. Add Positional encoding to patch embeddings 
    seq_len = cnn_tokens.shape[1]
    cnn_tokens = PositionalEncoding(
        cnn_tokens,
        seq_len=seq_len,
        emb_size=emb_size,
        drop_rate=attn_drop_rate,
    )

    # 3. Transformer encoder embedding enhancement through attention 
    att_tokens = TransformerEncoder(
        cnn_tokens,
        num_layers=num_layers, 
        emb_size=emb_size,
        num_heads=num_heads,
        att_drop=attn_drop_rate 
    )

    # 4. Skip connection embeddins (cnn + attention)
    features = Add()([cnn_tokens, att_tokens])

    # 5. Classification 
    x = Flatten()(features)
    x = Dropout(final_drop_rate)(x)
    outputs = Dense(nb_classes, activation="softmax")(x)

    return Model(inputs, outputs)



#################
# CUSTOM MODELS #
################# 
# Walter, Josu et. al 


# -------------------------------------------------------------------------
# Multi Scale Patch Embedding 

# This multiple kernel size patch embedding learns multiple
# temporal scales (multiple receptive fields), because EEG signals 
# are multi-scale by nature, this is learning the powerbanks without explicit
# preprocessing and its also a STRICT SUPERSET of CTNet (EEGNet style) embeddings
# 
# Being L the kernel length (temporal samples) and Fs the sampling frequency of
# the EEG signal, Period = L / Fs. If we want to capture the different waves
# atleast multiple periods of that wave has to be captured (1-3 periods). Kernel
# of size 3 will capture 12 ms (At Fs=250hz, each time sample is 4 ms), 
# then it will be able to capture fast gammas 100hz for example. Long filters will
# detect low frequencies and short filters will detect high frequencies. Obviously 
# large kernels can detect all as we said its a strict superset. Using smaller
# filters ,in addition, which add almost no weights to CNN and can enhance wave detection
# can be benefitial 
#  
# Delta waves 0.5-4 hz (Long kernel size!: 125)
# Theta waves 4-7 hz  (63)
# Alpha waves 8-12 hz (31)
# Beta waves 12 - 30hz (7 - 15)
# Gamma waves 30 - 100 hz (Short kernel size!: 3)

# It should be researched which kernel sizes and which filters per scale is optimum for eeg
# -------------------------------------------------------------------------

def MultiScalePatchEmbedding(
    x,
    Chans, 
    kernel_sizes=(5, 12, 25, 64 ,125),
    filters_per_scale=(8, 16, 16, 16, 4),
    pool_size=8, 
    final_dropout=0.3,
    spatial_dropout=0.1,
):
    branches = []
    for k, f in zip(kernel_sizes, filters_per_scale):
        b = Conv2D(f, (1, k), padding="same", use_bias=False)(x)
        b = BatchNormalization()(b)
        # Activation only after normalization avoids distortion of distrib. and stability
        b = Activation("elu")(b)
        branches.append(b)

    # Form a complex representation 
    x = Concatenate(axis=-1)(branches)

    # Spatial Filtering (Channels)
    x = DepthwiseConv2D((Chans, 1), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = SpatialDropout2D(spatial_dropout)(x)

    # Tokenization: Temporal Pooling into patches (B, T, D)
    x = AveragePooling2D((1, pool_size))(x)
    x = Dropout(final_dropout)(x)
    x = Reshape((-1, x.shape[-1]))(x)
    return x 


# The proposed Dual-Axis EEG Conformer explicitly models both temporal and channel-wise dependencies via axis-aware self-attention
# followed by cross-attention fusion, leading to improved representational capacity over CTNet
# This is in fact a refinement of CTNet and EEGConformer by building new layers on top of it, with stronger INDUCTIVE BIASES
# 1) multi-scale temporal CNN (e.g. 3, 7, 15, 31, 63, 125) to capture theta/alpha/beta bands better
# 2) Apply spatial CNN (electrodes correlations) AFTER temporal CNN (neurological inductive bias). LATE FUSION OF CNN + TRANSFORMER
# 3) Apply transformer attention computation to EEG channels too, so TemporalTransformer and ChannelTransformer are needed (2)
# 4) Cross attention fusion of the 2 Temporal/Channel encodings to allow capturing attention in channels too
# 5) Improve pooling mechanisms (learnable pooling + global pooling at the end)
# 6) Attention regularization || At A - I || 

# TODO: Improve positional embedding (Frequency aware with bias variable?)
def DualAttentionEEGConformer(
    # Basic
    nb_classes,
    Chans=64,
    Samples=256,
    emb_size=40,
    # CNN Patches 
    patch_kernel_sizes=(5, 12, 25, 64 ,125),
    patch_filters_per_scale=(8, 16, 16, 16, 4),
    patch_spatial_droprate=0.3,
    patch_final_droprate=0.1,
    patch_pool_size=8,
    # Positional encoding 
    pe_droprate=0.1,
    # Self-Attention Transformers
    time_num_heads=8,
    chan_num_heads=8,
    time_num_layers=4,
    chan_num_layers=4,
    time_att_droprate=0.1,
    chan_att_droprate=0.1,
    # Cross-Attention Transformer 
    cross_time_num_heads=8,
    cross_time_att_droprate=0.1,
    cross_chan_num_heads=8,
    cross_chan_att_droprate=0.1,
    # Final Pooling
    pool_droprate=0.4,
    # Classification Head & Pooling
    classif_hidden_units=(128, 32),
    classif_droprate_probs=(0.4, 0.2),    
) -> Model:

    # 0. Raw EEG signal, NumChannels * Time samples
    inp = Input((Chans, Samples, 1))

    # 1. Multi-scale CNN Patch Tokens
    tokens = MultiScalePatchEmbedding(
        inp, Chans, 
        patch_kernel_sizes,
        patch_filters_per_scale,
        patch_pool_size,
        patch_spatial_droprate,
        patch_final_droprate,
    )
    # Maybe an assertion of sum of filters >= embedding size could be useful here
    #assert(sum(patch_filters_per_scale) >= emb_size)

    # Analogous to ViT, add linear projection layer to avoid shape issues
    # the shape of the token depends on the sum of filters per scale

    tokens = Dense(emb_size, use_bias=False)(tokens)
    # Scale embeddings (standard transformer trick)
    tokens = tokens * tf.math.sqrt(tf.cast(emb_size, tf.float32))

    # 2. Add Positional encoding to patch embeddings 
    # TODO: Does this make sense to add?
    seq_len = tokens.shape[1]
    tokens = PositionalEncoding(
        tokens,
        seq_len=seq_len,
        emb_size=emb_size,
        drop_rate=pe_droprate,
    )

    # TODO: Check the shapes of embeddings and tokens they are mismatched right now
    # what is the size of the tokens?????? depends on the number of kernels right? 
    # here some pooling or dense layer should be added to perform dimensionality reduction 

    # 3. Temporal Transformer self-attention (B, T, D)
    time_embeddings = TransformerEncoder(
        tokens,
        emb_size=emb_size,
        num_layers=time_num_layers,
        num_heads=time_num_heads,
        att_drop=time_att_droprate,
    ) 

    # TODO: Solve this mess of channel embeddings  
    channel_embeddings = Permute((2, 1))(tokens)
    
    # 4. Channel Transformer self-attention 
    channel_embeddings = TransformerEncoder(
        channel_embeddings,
        emb_size=emb_size,
        num_layers=chan_num_layers,
        num_heads=chan_num_heads,
        att_drop=chan_att_droprate,
    )
    # Bring back to (B, T, D) for fusion
    channel_embeddings = Permute((2, 1))(channel_embeddings)


    # TODO: Cross-attention is incorrect because both embeddings shapes
    # are different time_x: (B, T, D) chan_x: (B, C, D)

    # 5. Cross-Attention fusion of Channel/Temporal embeddings
    time_fused = MultiHeadAttention(
        num_heads=cross_time_num_heads,
        key_dim=emb_size // cross_time_num_heads,
        dropout=cross_time_att_droprate
    )(
        query=time_embeddings,
        key=channel_embeddings,
        value=channel_embeddings
    )


    channel_fused = MultiHeadAttention(
        num_heads=cross_chan_num_heads,
        key_dim=emb_size // cross_chan_num_heads, 
        dropout=cross_chan_att_droprate,
    )

    x = Add()([time_fused, channel_fused, tokens])

    # 6. Cross-Attention pooling 
    x = LayerNormalization()(x)

    # GAP its okay but we could change it for attention pooling 
    # or pyramid pooling like presented below (presence + saliency)
    avg = GlobalAveragePooling1D()(x)
    max = GlobalMaxPooling1D()(x)

    x = Concatenate()([avg, max])

    x = Dropout(pool_droprate)(x)

    # 7. Classification head (Returns logits)
    out = ClassificationHead(
        x,
        nb_classes=nb_classes,
        hidden_units=classif_hidden_units,
        drop_probs=classif_droprate_probs, 
        activation="elu",
        use_batchnorm=False
    )

    return Model(inp, out)
