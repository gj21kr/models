import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, concatenate, BatchNormalization 
from tensorflow.keras.layers import Conv3D, UpSampling3D, Conv3DTranspose
from tensorflow.keras.layers import add
from tensorflow.keras.layers import LeakyReLU, Reshape, Lambda
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras
import numpy as np

def myConv(x_in, nf, strides=1, kernel_size = 3):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    x_out = Conv3D(nf, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = BatchNormalization()(x_out)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out


def UnetUpsample(l, num_filters):
    l = UpSampling3D()(l)
    l = myConv(l, num_filters)
    return l

def Unet3D(vol_size, depth=4, num_class=1, base_filter=32, activation='sigmoid'):
    inputs = Input(shape=vol_size)
    filters = []
    down_list = []
    deep_supervision = None
    layer = myConv(inputs, base_filter)
    
    for d in range(depth):
        num_filters = base_filter * (2**d)

        filters.append(num_filters)

        down_list.append(layer)
        if d != depth - 1:
            layer = myConv(layer, num_filters*2, strides=2)
        
    for d in range(depth-2, -1, -1):
        layer = UnetUpsample(layer, filters[d])
        layer = concatenate([layer, down_list[d]])
        layer = myConv(layer, filters[d])
        layer = myConv(layer, filters[d], kernel_size = 1)
        
        if deep_supervision:
            if 0< d < 3:
                pred = myConv(layer, num_class)
                if deep_supervision is None:
                    deep_supervision = pred
                else:
                    deep_supervision = add([pred, deep_supervision])
                deep_supervision = UpSampling3D()(deep_supervision)
    
    layer = myConv(layer, num_class, kernel_size = 1)
    
    if deep_supervision:
        layer = add([layer, deep_supervision])
    layer = Conv3D(num_class, kernel_size = 1)(layer)
    x = Activation(activation, name=activation)(layer)
#     pri
    model = Model(inputs=[inputs], outputs=[x])
    return model