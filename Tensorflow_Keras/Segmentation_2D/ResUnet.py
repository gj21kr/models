import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.regularizers import l2

def double_conv(inputs, filters, layer_activation='relu', padding='same'):    
    conv = Conv2D(filters, (3, 3), activation=layer_activation, padding=padding)(inputs)
    conv = Conv2D(filters, (3, 3), activation=layer_activation, padding=padding)(conv)
    conc = concatenate([inputs, conv], axis=-1)
    pool = MaxPooling2D(pool_size=(2, 2))(conc)
    return conv, pool
    
def up_conv(conv2, conv1, filters, layer_activation='relu', padding='same'):
    up = concatenate([Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv2), conv1], axis=-1)
    conv = Conv2D(filters, (3, 3), activation=layer_activation, padding=padding)(up)
    conv = Conv2D(filters, (3, 3), activation=layer_activation, padding=padding)(conv)
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
    
    conv5 = Conv2D(base_filter*2*2*2*2, (3, 3), activation=layer_activation, padding=padding)(pool4)
    conv5 = Conv2D(base_filter*2*2*2*2, (3, 3), activation=layer_activation, padding=padding)(conv5)
    conc5 = concatenate([pool4, conv5], axis=-1)

    conc6 = up_conv(conc5, conv4, base_filter*2*2*2, layer_activation=layer_activation, padding=padding)
    conc7 = up_conv(conc6, conv3, base_filter*2*2, layer_activation=layer_activation, padding=padding)
    conc8 = up_conv(conc7, conv2, base_filter*2, layer_activation=layer_activation, padding=padding)
    conc9 = up_conv(conc8, conv1, base_filter, layer_activation=layer_activation, padding=padding)

    conv10 = Conv2D(num_class, (1, 1), activation=activation)(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model

def ResUnet_multi(img_shape, base_filter=32, layer_activation='relu', padding='same', n_classes=1, n_classes_2=1):
    if n_classes == 1:
        activation='sigmoid'
    else:
        activation='softmax'
        
    inputs = Input((shape[0], shape[1], 1))
    
    conv1, pool1 = double_conv(inputs, filters=base_filter, layer_activation=layer_activation, padding=padding)
    conv2, pool2 = double_conv(pool1, filters=base_filter*2, layer_activation=layer_activation, padding=padding)
    conv3, pool3 = double_conv(pool2, filters=base_filter*2*2, layer_activation=layer_activation, padding=padding)
    conv4, pool4 = double_conv(pool3, filters=base_filter*2*2*2, layer_activation=layer_activation, padding=padding)
    
    ## first decoder
    conv5_1 = Conv2D(base_filter*2*2*2*2, (3, 3), activation=layer_activation, padding=padding)(pool4)
    conv5_1 = Conv2D(base_filter*2*2*2*2, (3, 3), activation=layer_activation, padding=padding)(conv5_1)
    conc5_1 = concatenate([pool4, conv5_1], axis=-1)

    conc6_1 = up_conv(conc5_1, conv4, base_filter*2*2*2, layer_activation=layer_activation, padding=padding)
    conc7_1 = up_conv(conc6_1, conv3, base_filter*2*2, layer_activation=layer_activation, padding=padding)
    conc8_1 = up_conv(conc7_1, conv2, base_filter*2, layer_activation=layer_activation, padding=padding)
    conc9_1 = up_conv(conc8_1, conv1, base_filter, layer_activation=layer_activation, padding=padding)

    conv10_1 = Conv2D(n_classes, (1, 1), activation=activation)(conc9_1)

    ## second decoder
    conv5_2 = Conv2D(base_filter*2*2*2*2, (3, 3), activation=layer_activation, padding=padding)(pool4)
    conv5_2 = Conv2D(base_filter*2*2*2*2, (3, 3), activation=layer_activation, padding=padding)(conv5_2)
    conc5_2 = concatenate([pool4, conv5_2], axis=-1)

    conc6_2 = up_conv(conc5_1, conv4, base_filter*2*2*2, layer_activation=layer_activation, padding=padding)
    conc7_2 = up_conv(conc6_1, conv3, base_filter*2*2, layer_activation=layer_activation, padding=padding)
    conc8_2 = up_conv(conc7_1, conv2, base_filter*2, layer_activation=layer_activation, padding=padding)
    conc9_2 = up_conv(conc8_1, conv1, base_filter, layer_activation=layer_activation, padding=padding)

    conv10_2 = Conv2D(n_classes_2, (1, 1), activation=activation)(conc9_2)
    
    model = Model(inputs=[inputs], outputs=[conv10_1,conv10_2])
    return model
