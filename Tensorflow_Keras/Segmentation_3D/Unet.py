import tensorflow as tf
from tensorflow.keras import Input, Model, layers, utils

from .utils import *

def Unet3D(shape, num_class=1, filters=32, model_depth=3, activation='sigmoid', pooling='max', layer_act='relu', bn_axis=-1):    
    nb_filter = [filters*2**i for i in range(5)]
    img_input = Input(shape=shape, name='main_input')

    conv1_1 = standard_unit(img_input, stage=model_depth-3, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    pool1 = pooling_unit(conv1_1, stage=model_depth-3, pooling=pooling)

    conv2_1 = standard_unit(pool1, stage=model_depth-2, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)
    pool2 = pooling_unit(conv2_1, stage=model_depth-2, pooling=pooling)

    conv3_1 = standard_unit(pool2, stage=model_depth-1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    pool3 = pooling_unit(conv3_1, stage=model_depth-1, pooling=pooling)

    conv4_1 = standard_unit(pool3, stage=model_depth, nb_filter=nb_filter[model_depth], layer_act=layer_act)

    _, output = decoder([conv1_1, conv2_1, conv3_1, conv4_1], nb_filter, decoder_depth=model_depth, decoder_num=0, layer_act=layer_act, activation=activation)

    return Model(inputs=img_input, outputs=output)


# https://arxiv.org/pdf/1703.07523.pdf
def Unet3D_MultiOutputs(shape, num_class=1, filters=32, first=4, end=3, model_depth=3, activation='sigmoid', layer_act='relu', pooling='max'):    
    nb_filter = [filters*2**i for i in range(5)]
    img_input = Input(shape=shape, name='main_input')
    conv1_1 = standard_unit(img_input, stage=model_depth-3, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    pool1 = pooling_unit(conv1_1, stage=model_depth-3, pooling=pooling)

    conv2_1 = standard_unit(pool1, stage=model_depth-2, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)
    pool2 = pooling_unit(conv2_1, stage=model_depth-2, pooling=pooling)

    conv3_1 = standard_unit(pool2, stage=model_depth-1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    pool3 = pooling_unit(conv3_1, stage=model_depth-1, pooling=pooling)

    conv4_1 = standard_unit(pool3, stage=model_depth, nb_filter=nb_filter[model_depth], layer_act=layer_act)
    
    outputs, last_out = decoder([conv1_1, conv2_1, conv3_1, conv4_1], nb_filter, decoder_depth=model_depth, decoder_num=0, layer_act=layer_act, activation=activation)
    
    conv2_1 = layers.UpSampling3D(size=2**1, name='out_2')(conv2_1)
    conv3_1 = layers.UpSampling3D(size=2**2, name='out_3')(conv3_1)
    conv4_1 = layers.UpSampling3D(size=2**3, name='out_4')(conv4_1)
    
    conv2_4 = layers.UpSampling3D(size=2**1, name='out_6')(outputs[1])
    conv1_5 = layers.UpSampling3D(size=2**2, name='out_7')(outputs[0])
    
    if first==1 and end==1:
        return Model(inputs=img_input, outputs=[conv1_1, last_out])
    elif first==2 and end==2:
        return Model(inputs=img_input, outputs=[conv1_1, conv2_1, conv2_4, last_out])    
    elif first==3 and end==3:
        return Model(inputs=img_input, outputs=[conv1_1, conv2_1, conv3_1, conv1_5, conv2_4, last_out])
    elif first==4 and end==3:
        return Model(inputs=img_input, outputs=[conv1_1, conv2_1, conv3_1, conv4_1, conv1_5, conv2_4, last_out])
    elif first==0 and end==1:
        return Model(inputs=img_input, outputs=[last_out])
    elif first==0 and end==2:
        return Model(inputs=img_input, outputs=[conv2_4, last_out])
    elif first==0 and end==3:
        return Model(inputs=img_input, outputs=[conv3_3, conv2_4, last_out])

    
def Unet3D_MultiDecoders(shape, num_decoders=2, num_class=1, filters=32, model_depth=3, decoder_num=2, activation='sigmoid', layer_act='relu', pooling='max'):    
    nb_filter = [filters*2**i for i in range(5)]
    img_input = Input(shape=shape, name='main_input')
    conv1_1 = standard_unit(img_input, stage=model_depth-3, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    pool1 = pooling_unit(conv1_1, stage=model_depth-3, pooling=pooling)

    conv2_1 = standard_unit(pool1, stage=model_depth-2, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)
    pool2 = pooling_unit(conv2_1, stage=model_depth-2, pooling=pooling)

    conv3_1 = standard_unit(pool2, stage=model_depth-1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    pool3 = pooling_unit(conv3_1, stage=model_depth-1, pooling=pooling)

    conv4_1 = standard_unit(pool3, stage=model_depth, nb_filter=nb_filter[model_depth], layer_act=layer_act)

    unet_outputs = []
    for decoder_num in range(num_decoders):
        _, last_out = decoder([conv1_1, conv2_1, conv3_1, conv4_1], nb_filter, decoder_depth=model_depth, decoder_num=decoder_num, layer_act=layer_act, activation=activation)
        unet_outputs.append(last_out)

    return Model(inputs=img_input, outputs=unet_outputs)

