# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:04:00 2020

@author: JEpark
"""
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Input, Conv2DTranspose
from tensorflow.keras.layers import concatenate, LayerNormalization
from tensorflow.keras.models import Model
#     from keras_layer_normalization import LayerNormalization


def conv_block(x, filters, kernel_size=(3,3), padding='same', mode_norm='batch', data_format='channels_last'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, padding=padding, kernel_initializer='he_normal')(x)
    if mode_norm=='batch':
        x = BatchNormalization()(x)
    else :
        x = LayerNormalization()(x)
    x = Activation('relu')(x)
    return x

def deconv_block(x, skips, filters, kernel_size=(3,3), strides=(2,2), mode_norm='batch', padding='same', data_format='channels_last'):
    x = Conv2DTranspose(filters=filters, kernel_size=(2,2), strides=strides, padding=padding)(x)
    x = concatenate([skips, x], axis=3)    
    x = conv_block(x, filters=filters, mode_norm=mode_norm)
    x = conv_block(x, filters=filters, mode_norm=mode_norm)
    
    return x

#####################################################################################################################################

def return_activation(n_shapes):
    if n_shapes==1:
        return 'sigmoid'
    else:
        return 'softmax'
    
#U-Net
def unet(shape, num_class, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(shape)
    x = inputs    
    
    activation = return_activation(num_class)
    
    skips = []
    for i in range(depth):
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = deconv_block(x, skips[i], features, data_format=data_format)

    conv6 = Conv2D(num_class, (1, 1), padding='same', data_format=data_format, activation=activation)(x)
    model = Model(inputs=inputs, outputs=conv6)

    return model

# Unet w/ multiple branch
def unet_edge(shape, num_class, num_class2, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(shape)
    
    activation = return_activation(num_class)    
    activation_e = return_activation(num_class2)
    
    x = inputs
    skips = []
    for i in range(depth):
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)

        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)

    x1 = x
    for i in reversed(range(depth)):
        features = features // 2
        x1 = deconv_block(x1, skips[i], features, data_format=data_format)
    
    conv6 = Conv2D(num_class, (1, 1), padding='same', data_format=data_format, activation=activation)(x1)

    x2 = x
    for i in reversed(range(depth)):
        features = features // 2
        x2 = deconv_block(x2, skips[i], features, data_format=data_format)
        
    conv8 = Conv2D(n_edges, (1, 1), padding='same', data_format=data_format, activation=activation_e)(x2)

    model = Model(num_class2=inputs, outputs=[conv6, conv8])

    return model

# Unet w/ multiple branch w/ multi-inputs w/o deep-supervision
def unet_edge_avg(shape, num_class, num_class2, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(shape)
    x = inputs
    
    activation = return_activation(num_class)  
    activation_e = return_activation(num_class2)  

    avg = inputs
    avgs = []
    for i in range(depth):
        avg = AveragePooling2D(pool_size=(2, 2))(avg)
        avgs.append(avg)

    skips = []
    for i in range(depth):
        
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        avg2 = conv_block(avgs[i], features, mode_norm='batch', data_format=data_format)
        x = concatenate([x, avg2])

        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)

    x1 = x
    for i in reversed(range(depth)):
        features = features // 2
        x1 = deconv_block(x1, skips[i], features, mode_norm=mode_norm, data_format=data_format)
    
    conv6 = Conv2D(num_class, (1, 1), padding='same', data_format=data_format, activation=activation)(x1)

    x2 = x
    for i in reversed(range(depth)):
        features = features // 2
        x2 = deconv_block(x2, skips[i], features, data_format=data_format)
        
    conv8 = Conv2D(num_class2, (1, 1), padding='same', data_format=data_format, activation=activation_e)(x2)

    model = Model(inputs=inputs, outputs=[conv6, conv8])

    return model


# Unet w/o multiple branch w/ multi-inputs w/ supervision 
def unet_sv(shape, num_class, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(shape)
    x = inputs

    activation = return_activation(num_class) 
    
    avg = inputs
    avgs = []
    for i in range(depth):
        avg = AveragePooling2D(pool_size=(2, 2))(avg)
        avgs.append(avg)

    skips = []
    for i in range(depth):
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        avg2 = conv_block(avgs[i], features, mode_norm=mode_norm, data_format=data_format)
        x = concatenate([x, avg2])

        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)

    outs = []
    up_size = depth * 2
    for i in reversed(range(depth)):
        features = features // 2
        x = deconv_block(x, skips[i], features, mode_norm=mode_norm, data_format=data_format)
    
        conv_out = UpSampling2D(size=(up_size, up_size))(x)
        outs.append(Conv2D(num_class, (3,3), activation=activation, padding='same', name='out_{}'.format(i))(conv_out))
        up_size = up_size // 2
        
    model = Model(inputs=inputs, outputs=outs)

    return model



# Unet w multiple branch w/ multi-inputs w/ supervision 
def unet_sv(shape, num_class, num_class2, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(shape)
    x = inputs

    activation = return_activation(num_class) 
    activation = return_activation(num_class2) 
    
    avg = inputs
    avgs = []
    for i in range(depth):
        avg = AveragePooling2D(pool_size=(2, 2))(avg)
        avgs.append(avg)

    skips = []
    for i in range(depth):
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        avg2 = conv_block(avgs[i], features, mode_norm=mode_norm, data_format=data_format)
        x = concatenate([x, avg2])

        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)

    x1 = x
    x2 = x
    outs = []
    up_size = depth * 2
    for i in reversed(range(depth)):
        features = features // 2
        x1 = deconv_block(x1, skips[i], features, mode_norm=mode_norm, data_format=data_format)
    
        conv_out = UpSampling2D(size=(up_size, up_size))(x1)
        outs.append(Conv2D(num_class, (3,3), activation=activation, padding='same', name='out_{}'.format(i))(conv_out))
        up_size = up_size // 2
        
    for i in reversed(range(depth)):
        features = features // 2
        x2 = deconv_block(x2, skips[i], features, mode_norm=mode_norm, data_format=data_format)
    
        conv_out = UpSampling2D(size=(up_size, up_size))(x2)
        outs.append(Conv2D(num_class2, (3,3), activation=activation, padding='same', name='out_{}'.format(i))(conv_out))
        up_size = up_size // 2
        
    model = Model(inputs=inputs, outputs=outs)

    return model


# Unet w/ supervision 
def unet_sv_outSz(in_shape, out_shape, n_shapes, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(in_shape)
    size = out_shape // int(in_shape[0])

    activation = return_activation(n_shapes) 
    
    x = inputs

    avg = inputs
    avgs = []
    for i in range(depth):
        avg = AveragePooling2D(pool_size=(2, 2))(avg)
        avgs.append(avg)

    skips = []
    for i in range(depth):        
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        avg2 = conv_block(avgs[i], features, mode_norm=mode_norm, data_format=data_format)
        x = concatenate([x, avg2])
        
        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)

    outs = []
    up_size = depth * (2 * size)
    for i in reversed(range(depth)):
        features = features // 2
        x = deconv_block(x, skips[i], features, mode_norm=mode_norm, data_format=data_format)    
        conv_out = UpSampling2D(size=(up_size, up_size))(x)
        outs.append(Conv2D(n_shapes, 3, activation=activation, padding='same')(conv_out))
        up_size = up_size // 2
                
    model = Model(inputs=inputs, outputs=outs)

    return model
