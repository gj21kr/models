# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:01:09 2020

@author: JEpark
"""


from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Input, Dropout
from keras.layers import core, add, multiply
from keras.models import Model

from keras.layers.core import Lambda
import keras.backend as K

def attention_up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate

def attention_block_2d(x, g, inter_channel, data_format='channels_first'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x

#Attention U-Net
def att_unet(img_w, img_c, num_shapes, depth = 4, features = 64, data_format='channels_first'):
    inputs = Input((img_w, img_w, img_c))
    x = inputs
    
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(num_shapes, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    model.summary()
    return model

#Attention U-Net with edge output
def att_unet_edge(img_w, img_c, num_shapes, num_edges, depth = 4, features = 64, data_format='channels_first'):
    inputs = Input((img_w, img_w, img_c))
    x = inputs

    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    x1 = x
    for i in reversed(range(depth)):
        features = features // 2
        x1 = attention_up_and_concate(x1, skips[i], data_format=data_format)
        x1 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x1)
        x1 = Dropout(0.2)(x1)
        x1 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x1)
        
    conv1 = Conv2D(num_shapes, (1, 1), padding='same', data_format=data_format)(x1)
    conv2 = core.Activation('sigmoid')(conv1)
    
    x2 = x
    for i in reversed(range(depth)):
        features = features // 2
        x2 = attention_up_and_concate(x2, skips[i], data_format=data_format)
        x2 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x2)
        x2 = Dropout(0.2)(x2)
        x2 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x2)

    conv3 = Conv2D(num_edges, (1, 1), padding='same', data_format=data_format)(x2)
    conv4 = core.Activation('sigmoid')(conv3)
    
    model = Model(inputs=inputs, outputs=[conv2, conv4])

    model.summary()
    return model