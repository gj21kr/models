import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.regularizers import l2

def double_conv(inputs, filters, layer_activation='relu', padding='same'):    
    conv = Conv3D(filters, 3, activation=layer_activation, padding=padding)(inputs)
    conv = Conv3D(filters, 3, activation=layer_activation, padding=padding)(conv)
    conc = concatenate([inputs, conv], axis=-1)
    pool = MaxPooling3D(pool_size=2)(conc)
    return conv, pool
    
def up_conv(conv2, conv1, filters, layer_activation='relu', padding='same'):
    up = concatenate([Conv2DTranspose(filters, 2, strides=2, padding='same')(conv2), conv1], axis=-1)
    conv = Conv3D(filters, 3, activation=layer_activation, padding=padding)(up)
    conv = Conv3D(filters, 3, activation=layer_activation, padding=padding)(conv)
    conc = concatenate([up, conv], axis=-1)
    return conc
    
    
def ResUnet(shape, base_filter=32, layer_activation='relu', padding='same', num_class=1):
    if n_classes == 1:
        activation='sigmoid'
    else:
        activation='softmax'
        
    inputs = Input(shape)
    
    conv1, pool1 = double_conv(inputs, filters=base_filter, layer_activation=layer_activation, padding=padding)
    conv2, pool2 = double_conv(pool1, filters=base_filter*2, layer_activation=layer_activation, padding=padding)
    conv3, pool3 = double_conv(pool2, filters=base_filter*2*2, layer_activation=layer_activation, padding=padding)
    conv4, pool4 = double_conv(pool3, filters=base_filter*2*2*2, layer_activation=layer_activation, padding=padding)
    
    conv5 = Conv3D(base_filter*2*2*2*2, 3, activation=layer_activation, padding=padding)(pool4)
    conv5 = Conv3D(base_filter*2*2*2*2, 3, activation=layer_activation, padding=padding)(conv5)
    conc5 = concatenate([pool4, conv5], axis=-1)

    conc6 = up_conv(conc5, conv4, base_filter*2*2*2, layer_activation=layer_activation, padding=padding)
    conc7 = up_conv(conc6, conv3, base_filter*2*2, layer_activation=layer_activation, padding=padding)
    conc8 = up_conv(conc7, conv2, base_filter*2, layer_activation=layer_activation, padding=padding)
    conc9 = up_conv(conc8, conv1, base_filter, layer_activation=layer_activation, padding=padding)

    conv10 = Conv23(num_class, (1, 1), activation=activation)(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model