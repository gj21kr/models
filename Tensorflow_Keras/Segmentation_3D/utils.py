import tensorflow as tf
from tensorflow.keras import Input, Model, layers, utils

def standard_unit(input_tensor, stage, nb_filter, layer_act, kernel_size=3):
    x = layers.Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), activation=None, name='conv'+str(stage)+'_1', kernel_initializer = 'he_normal', padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    if layer_act == 'relu':
        x = layers.ReLU()(x)
    elif layer_act == 'leaky_relu':
        x = layers.LeakyReLU()(x)
    x = layers.Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), activation=None, name='conv'+str(stage)+'_2', kernel_initializer = 'he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    if layer_act == 'relu':
        x = layers.ReLU()(x)
    elif layer_act == 'leaky_relu':
        x = layers.LeakyReLU()(x)
    return x

def pooling_unit(conv1_1, stage, pooling='max', kernel_size=2, stride=2):
    if pooling=='max':
        return layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool'+str(stage))(conv1_1)
    elif pooling=='avg':
        return layers.AveragePooling3D((2, 2, 2), strides=(2, 2, 2), name='pool'+str(stage))(conv1_1)
    else:
        return conv1_1
    
def decoder(convs, nb_filter, decoder_depth=3, decoder_num=0, layer_act='relu', activation='sigmoid', bn_axis=-1):  
    outputs = []
    start_layer = convs[-1]
    for idx in range(decoder_depth):
        up = layers.UpSampling3D(name='up'+str(idx)+str(decoder_num))(start_layer)
        conv = layers.concatenate([up, convs[-(idx+2)]], name='merge'+str(idx)+str(decoder_num), axis=bn_axis)
        conv = standard_unit(conv, stage=str(idx)+str(decoder_num), nb_filter=nb_filter[2], layer_act=layer_act)
        start_layer = conv
        outputs.append(conv)

    conv_last = layers.Conv3D(1, (1, 1, 1), activation=activation, name='output_'+str(decoder_num), kernel_initializer='he_normal', padding='same')(start_layer)

    return outputs, conv_last
    
        
def decoder_att_321(conv1_1, conv2_1, conv3_1, conv4_1, decoder_num=0, layer_act='relu', bn_axis=-1):    
    up3_3 = layers.UpSampling3D(name='up33_'+str(decoder_num))(conv4_1)
    #up3_3 = layers.Conv3DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    att3_3 = layers.Attention()([up3_3, conv3_1])
    conv3_3 = layers.concatenate([up3_3, att3_3], name='merge33_'+decoder_num, axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33_'+str(decoder_num), nb_filter=nb_filter[2], layer_act=layer_act)

    up2_4 = layers.UpSampling3D(name='up24_'+str(decoder_num))(conv3_3)
    #up2_4 = layers.Conv3DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    att2_4 = layers.Attention()([up2_4, conv2_1])
    conv2_4 = layers.concatenate([up2_4, att2_4], name='merge24_'+str(decoder_num), axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24_'+str(decoder_num), nb_filter=nb_filter[1], layer_act=layer_act)

    up1_5 = layers.UpSampling3D(name='up15_'+str(decoder_num))(conv2_4)
    #up1_5 = layers.Conv3DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    att1_5 = layers.Attention()([up1_5, conv1_1])
    conv1_5 = layers.concatenate([up1_5, att1_5], name='merge15_'+str(decoder_num), axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15_'+str(decoder_num), nb_filter=nb_filter[0], layer_act=layer_act)

    unet_output = layers.Conv3D(1, (1, 1, 1), activation=activation, name='output_'+str(decoder_num), kernel_initializer = 'he_normal', padding='same')(conv1_5)
    return unet_output

def decoder_att_21(conv1_1, conv2_1, conv3_1, conv4_1, decoder_num=0, layer_act='relu', bn_axis=-1):    
    up3_3 = layers.UpSampling3D(name='up33_'+str(decoder_num))(conv4_1)
    #up3_3 = layers.Conv3DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = layers.concatenate([up3_3, conv3_1], name='merge33_'+str(decoder_num), axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33_'+str(decoder_num), nb_filter=nb_filter[2], layer_act=layer_act)

    up2_4 = layers.UpSampling3D(name='up24_'+decoder_num)(conv3_3)
    #up2_4 = layers.Conv3DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    att2_4 = layers.Attention()([up2_4, conv2_1])
    conv2_4 = layers.concatenate([up2_4, att2_4], name='merge24_'+str(decoder_num), axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24_'+str(decoder_num), nb_filter=nb_filter[1], layer_act=layer_act)

    up1_5 = layers.UpSampling3D(name='up15_'+decoder_num)(conv2_4)
    #up1_5 = layers.Conv3DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    att1_5 = layers.Attention()([up1_5, conv1_1])
    conv1_5 = layers.concatenate([up1_5, att1_5], name='merge15_'+str(decoder_num), axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15_'+str(decoder_num), nb_filter=nb_filter[0], layer_act=layer_act)

    unet_output = layers.Conv3D(1, (1, 1, 1), activation=activation, name='output_'+str(decoder_num), kernel_initializer = 'he_normal', padding='same')(conv1_5)
    return unet_output


def decoder_att_32(conv1_1, conv2_1, conv3_1, conv4_1, decoder_num=0, layer_act='relu', bn_axis=-1):    
    up3_3 = layers.UpSampling3D(name='up33_'+str(decoder_num))(conv4_1)
    #up3_3 = layers.Conv3DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    att3_3 = layers.Attention()([up3_3, conv3_1])
    conv3_3 = layers.concatenate([up3_3, att3_3], name='merge33_'+str(decoder_num), axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33_'+str(decoder_num), nb_filter=nb_filter[2], layer_act=layer_act)

    up2_4 = layers.UpSampling3D(name='up24_'+str(decoder_num))(conv3_3)
    #up2_4 = layers.Conv3DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    att2_4 = layers.Attention()([up2_4, conv2_1])
    conv2_4 = layers.concatenate([up2_4, att2_4], name='merge24_'+str(decoder_num), axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24_'+str(decoder_num), nb_filter=nb_filter[1], layer_act=layer_act)

    up1_5 = layers.UpSampling3D(name='up15_'+str(decoder_num))(conv2_4)
    #up1_5 = layers.Conv3DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = layers.concatenate([up1_5, conv1_1], name='merge15_'+str(decoder_num), axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15_'+str(decoder_num), nb_filter=nb_filter[0], layer_act=layer_act)

    unet_output = layers.Conv3D(1, (1, 1, 1), activation=activation, name='output_'+str(decoder_num), kernel_initializer = 'he_normal', padding='same')(conv1_5)
    return unet_output

