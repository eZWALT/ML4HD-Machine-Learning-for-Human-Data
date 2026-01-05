from typing import List
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  LSTM, GRU
from tensorflow.keras.layers import Conv1D, MaxPool1D, Add, Concatenate, Reshape
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, AveragePooling1D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D, SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Add, MultiHeadAttention
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2


def Simple_CNN(nb_classes, input_shape=(64, 128, 1)):
        
    input1 = Input(shape =input_shape)
    block1 = Conv2D(32, (3, 3), activation='elu', padding='same')(input1)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D((2, 1))(block1) # Only reduce height, keep time width

    # 2. Second Convolutional Block
    block2 = Conv2D(64, (3, 3), activation='elu', padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling2D((2, 1))(block2)

    # 3. Flatten and Classify
    block3 = Flatten()(block2)
    block3 = Dense(64, activation='elu')(block3)
    block3 = Dropout(0.5)(block3) # Prevent overfitting

    dense = Dense(nb_classes)(block3)
 
    softmax = Activation('softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


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
def PatchEmbedding(
    input_tensor,
    n_filters_time=40,
    filter_time_length=25,
    n_chans=64,
    pool_time_length=75,
    pool_time_stride=15,
    drop_prob=0.5
):
    # Temporal Convolution (1 x Lt)
    x = Conv2D(
        n_filters_time, 
        (1, filter_time_length),
        padding="valid",
        use_bias=True
    )(input_tensor)
    
    # Spatial Convolution (N Channels x 1)
    x = Conv2D(
        n_filters_time, 
        (n_chans, 1),
        padding="valid",
        use_bias=True,
    )(x)

    # Batch Normalization, activation and temporal pooling --> patches and dropout
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


def PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, embed_dim, drop_rate=0.1):
        super().__init__()
        self.pos_emb = self.add_weight(
            shape=(1, seq_len, embed_dim),
            initializer="random_normal",
            trainable=True, 
        )
        self.dropout = Dropout(drop_rate)

    def call(self, x, training=False):
        x = x + self.pos_emb[:, :tf.shape(x)[1], :]
        return self.dropout(x, training=training) 


############ COMMON MODULES

def EEGConformer(
    nb_classes, Chans = 64, Samples = 256, n_filters_time=40,
    filter_time_length=25,pool_time_length=75,  pool_time_stride=15,
    drop_prob=0.5, num_layers=6, num_heads=10, att_drop_prob=0.5
) -> Model:

    input_eeg = Input(shape=(Chans, Samples, 1))

    # 1. Patch embedding. Use a CNN to get embeddings from temporal windows
    # EEG signal (64 x 256) -> (CNN) -> 40 Filters -> (Pooling) -> T patches of 40-D vectors 
    x = PatchEmbedding(
        input_tensor=input_eeg,
        n_filters_time=n_filters_time, 
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

def CTNet(nb_classes, Chans = 64, Samples = 256):
    pass 