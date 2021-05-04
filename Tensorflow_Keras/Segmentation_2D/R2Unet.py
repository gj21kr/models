# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:06:14 2020

@author: JEpark
"""

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Input
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
from tensorflow.keras.layers.core import Lambda
import tensorflow.keras.backend as K

def up_and_concate(down_layer, layer, data_format='channels_last'):

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate

def res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1], padding='same', data_format='channels_last'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    layer = input_layer
    for i in range(2):
        layer = Conv2D(out_n_filters // 4, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(out_n_filters // 4, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
        layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    out_layer = add([layer, skip_layer])
    return out_layer


# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1], padding='same', data_format='channels_last'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):
        for i in range(2):
            if i == 0:
                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer

#Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def r2_unet(img_shape, n_classes, features=64, data_format='channels_last'):
    inputs = Input(img_shape)
    x = inputs
    depth = 4
    
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = Conv2D(n_classes, (1, 1), padding='same', data_format=data_format, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=conv6)

    return model

#Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def r2_unet_multi(img_shape, n_classes, features=64, data_format='channels_last'):
    inputs = Input(img_shape)
    x = inputs
    depth = 4
    
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x1 = rec_res_block(x, features, data_format=data_format)
    for i in reversed(range(depth)):
        features = features // 2
        x1 = up_and_concate(x1, skips[i], data_format=data_format)
        x1 = rec_res_block(x1, features, data_format=data_format)
    conv1 = Conv2D(n_classes, (1, 1), padding='same', data_format=data_format, activation='sigmoid')(x1)


    x2 = rec_res_block(x, features, data_format=data_format)
    for i in reversed(range(depth)):
        features = features // 2
        x2 = up_and_concate(x2, skips[i], data_format=data_format)
        x2 = rec_res_block(x2, features, data_format=data_format)
    conv2 = Conv2D(n_classes, (1, 1), padding='same', data_format=data_format, activation='sigmoid')(x2)
    
    model = Model(inputs=inputs, outputs=[conv1, conv2])
    return model