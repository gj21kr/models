import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Model, layers, utils

def standard_unit(input_tensor, stage, nb_filter, layer_act, kernel_size=3):
    x = layers.Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), activation=None, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    if layer_act == 'relu':
        x = layers.ReLU(x)
    elif layer_act == 'leaky_relu':
        x = layers.LeakyReLU()(x)
    x = layers.Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), activation=None, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    if layer_act == 'relu':
        x = layers.ReLU(x)
    elif layer_act == 'leaky_relu':
        x = layers.LeakyReLU()(x)
    return x

def pooling_unit(conv1_1, stage, pooling='max', kernel_size=2, stride=2):
    if pooling=='max':
        return layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool'+stage)(conv1_1)
    elif pooling=='avg':
        return layers.AveragePooling3D((2, 2, 2), strides=(2, 2, 2), name='pool'+stage)(conv1_1)
    else:
        return conv1_1

def down_unit(img_input, stage, num_filters = [9,17,27], pool=True):    
    conv1_a = standard_unit(img_input, stage=f'{stage}a', nb_filter=num_filters[0], layer_act=layer_act)
    conv1_b = standard_unit(conv1_a, stage=f'{stage}b', nb_filter=num_filters[1], layer_act=layer_act)
    conv1_c = standard_unit(conv1_b, stage=f'{stage}c', nb_filter=num_filters[2], layer_act=layer_act)
    conv1_d = layers.concatenate([conv1_c, conv1_b], name=f'merge{stage}b', axis=bn_axis)
    conv1_d = layers.concatenate([conv1_d, conv1_a], name=f'merge{stage}a', axis=bn_axis)
    conv1_A = layers.Conv3D(np.sum(num_filters), (1, 1, 1), activation=activation, name=f'stage{stage}', kernel_initializer = 'he_normal', padding='same')(img_input)
    conv1_1 = layers.Add()([conv1_A,conv1_d])
    if not pool:
        return conv1_1
    else:
        pool1 = pooling_unit(conv1_1, stage=stage,pooling=pooling)
        return conv1_1, pool1
    
def up_unit(conv4_1,conv3_1,stage):
    up3_3 = layers.UpSampling3D(name='up'+stage)(conv4_1)
    #up3_3 = layers.Conv3DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)    
    att3 = layers.Attention()([conv3_1, up3_3])    
    conv3_3 = layers.concatenate([up3_3, att3], name='merge'+stage, axis=bn_axis)
    return conv3_3

def Unet3D(shape, num_class=1, filters=32, activation='sigmoid', layer_act='relu', pooling='max'):    
    bn_axis = -1
    img_input = Input(shape=shape, name='main_input')

    conv1_1, pool1 = down_unit(img_input, 1, num_filters = [9,17,27])        
    conv2_1, pool2 = down_unit(pool1, 2, num_filters = [9,17,27])    
    conv3_1, pool3 = down_unit(pool2, 3, num_filters = [18,35,54])
    conv4_1 = down_unit(pool3, 3, num_filters = [36,70,108], pool=False)

    conv3_2 = up_unit(conv4_1, conv3_1, 3)
    conv2_2 = up_unit(conv3_2, conv2_1, 3)
    conv1_2 = up_unit(conv2_2, conv1_1, 3)

    unet_output = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='output', 
                                kernel_initializer = 'he_normal', padding='same')(conv1_2)
    
    return Model(inputs=img_input, outputs=unet_output)