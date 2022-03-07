import tensorflow as tf
from tensorflow.keras import Input, Model, layers, utils

def standard_unit(input_tensor, nb_filter, layer_act, kernel_size=3):
    x = layers.Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), activation=None, kernel_initializer = 'he_normal', padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    if layer_act == 'relu':
        x = layers.ReLU()(x)
    elif layer_act == 'leaky_relu':
        x = layers.LeakyReLU()(x)
    x = layers.Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), activation=None, kernel_initializer = 'he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    if layer_act == 'relu':
        x = layers.ReLU()(x)
    elif layer_act == 'leaky_relu':
        x = layers.LeakyReLU()(x)
    return x


def pooling_unit(conv1_1, pooling='max', kernel_size=2, stride=2):
    if pooling=='max':
        return layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1_1)
    elif pooling=='avg':
        return layers.AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(conv1_1)
    else:
        return conv1_1
    
def up_unit(conv4_1, conv3_1, nb_filter, bn_axis=-1, layer_act='relu'):
    up = layers.UpSampling3D()(conv4_1)
    conv = layers.concatenate([up, conv3_1], axis=bn_axis)
    conv = standard_unit(conv, nb_filter=nb_filter, layer_act=layer_act)
    return conv
    
def Unet3D(shape, num_classes=1, filters=32, model_depth=3, activation='sigmoid', pooling='max', layer_act='relu', bn_axis=-1):    
    nb_filter = [filters*2**i for i in range(5)]
    img_input = Input(shape=shape, name='main_input')

    conv1_1 = standard_unit(img_input, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    pool1 = pooling_unit(conv1_1, pooling=pooling)

    conv2_1 = standard_unit(pool1, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)
    pool2 = pooling_unit(conv2_1, pooling=pooling)

    conv3_1 = standard_unit(pool2, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    pool3 = pooling_unit(conv3_1, pooling=pooling)

    conv4_1 = standard_unit(pool3, nb_filter=nb_filter[model_depth], layer_act=layer_act)

    conv3_2 = up_unit(conv4_1, conv3_1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    conv2_2 = up_unit(conv3_2, conv2_1, nb_filter=nb_filter[model_depth-2], layer_act=layer_act) 
    conv1_2 = up_unit(conv2_2, conv1_1, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)

    output = layers.Conv3D(num_classes, (1, 1, 1), activation=activation, name='output', kernel_initializer='he_normal', padding='same')(conv1_2)

    return Model(inputs=img_input, outputs=output)

    
def Unet3D_MultiDecorders(shape, num_classes=1, filters=32, model_depth=3, activation='sigmoid', pooling='max', layer_act='relu', bn_axis=-1):    
    nb_filter = [filters*2**i for i in range(5)]
    img_input = Input(shape=shape, name='main_input')

    conv1_1 = standard_unit(img_input, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    pool1 = pooling_unit(conv1_1, pooling=pooling)

    conv2_1 = standard_unit(pool1, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)
    pool2 = pooling_unit(conv2_1, pooling=pooling)

    conv3_1 = standard_unit(pool2, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    pool3 = pooling_unit(conv3_1, pooling=pooling)

    conv4_1 = standard_unit(pool3, nb_filter=nb_filter[model_depth], layer_act=layer_act)

    conv3_2 = up_unit(conv4_1, conv3_1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    conv2_2 = up_unit(conv3_2, conv2_1, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)
    conv1_2 = up_unit(conv2_2, conv1_1, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)

    conv3_3 = up_unit(conv4_1, conv3_1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    conv2_3 = up_unit(conv3_3, conv2_1, nb_filter=nb_filter[model_depth-2], layer_act=layer_act) 
    conv1_3 = up_unit(conv2_3, conv1_1, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    
    conv3_4 = up_unit(conv4_1, conv3_1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act) 
    conv2_4 = up_unit(conv3_4, conv2_1, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)
    conv1_4 = up_unit(conv2_4, conv1_1, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    
    conv3_5 = up_unit(conv4_1, conv3_1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    conv2_5 = up_unit(conv3_5, conv2_1, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)  
    conv1_5 = up_unit(conv2_5, conv1_1, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    
    conv3_6 = up_unit(conv4_1, conv3_1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    conv2_6 = up_unit(conv3_6, conv2_1, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)   
    conv1_6 = up_unit(conv2_6, conv1_1, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
       
    
    output1 = layers.Conv3D(1, (1, 1, 1), activation=activation, name='output1', kernel_initializer='he_normal', padding='same')(conv1_2)
    output2 = layers.Conv3D(1, (1, 1, 1), activation=activation, name='output2', kernel_initializer='he_normal', padding='same')(conv1_3)
    output3 = layers.Conv3D(1, (1, 1, 1), activation=activation, name='output3', kernel_initializer='he_normal', padding='same')(conv1_4)
    output4 = layers.Conv3D(1, (1, 1, 1), activation=activation, name='output4', kernel_initializer='he_normal', padding='same')(conv1_5)
    output5 = layers.Conv3D(1, (1, 1, 1), activation=activation, name='output5', kernel_initializer='he_normal', padding='same')(conv1_6)

    output = layers.concatenate([output1, output2])
    output = layers.concatenate([output, output3])
    output = layers.concatenate([output, output4])
    output = layers.concatenate([output, output5])
    return Model(inputs=img_input, outputs=output)

# https://arxiv.org/pdf/1703.07523.pdf


