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
    
def up_unit(conv4_1, conv3_1, stage, nb_filter, bn_axis=-1, layer_act='relu'):
    up = layers.UpSampling3D(name='up_'+str(stage))(conv4_1)
    conv = layers.concatenate([up, conv3_1], name='merge_'+str(stage), axis=bn_axis)
    conv = standard_unit(conv, stage='up_last_'+str(stage), nb_filter=nb_filter, layer_act=layer_act)
    return conv
    
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

    conv3_2 = up_unit(conv4_1, conv3_1, stage=model_depth-1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)                             
    conv2_2 = up_unit(conv3_2, conv2_1, stage=model_depth-2, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)                             
    conv1_2 = up_unit(conv2_2, conv1_1, stage=model_depth-3, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)

    output = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='output', kernel_initializer='he_normal', padding='same')(conv1_2)

    return Model(inputs=img_input, outputs=output)

    
def Unet3D_MultiDecorders(shape, num_class=1, filters=32, model_depth=3, activation='sigmoid', pooling='max', layer_act='relu', bn_axis=-1):    
    nb_filter = [filters*2**i for i in range(5)]
    img_input = Input(shape=shape, name='main_input')

    conv1_1 = standard_unit(img_input, stage=model_depth-3, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    pool1 = pooling_unit(conv1_1, stage=model_depth-3, pooling=pooling)

    conv2_1 = standard_unit(pool1, stage=model_depth-2, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)
    pool2 = pooling_unit(conv2_1, stage=model_depth-2, pooling=pooling)

    conv3_1 = standard_unit(pool2, stage=model_depth-1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    pool3 = pooling_unit(conv3_1, stage=model_depth-1, pooling=pooling)

    conv4_1 = standard_unit(pool3, stage=model_depth, nb_filter=nb_filter[model_depth], layer_act=layer_act)

    conv3_2 = up_unit(conv4_1, conv3_1, stage=model_depth-1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)                             
    conv2_2 = up_unit(conv3_2, conv2_1, stage=model_depth-2, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)                             
    conv1_2 = up_unit(conv2_2, conv1_1, stage=model_depth-3, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)

    conv3_3 = up_unit(conv4_1, conv3_1, stage=model_depth-1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)                             
    conv2_3 = up_unit(conv3_3, conv2_1, stage=model_depth-2, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)                             
    conv1_3 = up_unit(conv2_3, conv1_1, stage=model_depth-3, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    
    output1 = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='output1', kernel_initializer='he_normal', padding='same')(conv1_2)
    output2 = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='output2', kernel_initializer='he_normal', padding='same')(conv1_3)

    return Model(inputs=img_input, outputs=[output1,output2])

# https://arxiv.org/pdf/1703.07523.pdf
def Unet3D_MultiOutputs(shape, num_class=1, filters=32, first=4, end=3, model_depth=3, activation='sigmoid', layer_act='relu', pooling='max'):    
    nb_filter = [filters*2**i for i in range(5)]
    img_input = Input(shape=shape, name='main_input')
    conv1_1 = standard_unit(img_input, stage=model_depth-3, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    conv1_2 = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='out_1', kernel_initializer='he_normal', padding='same')(conv1_1)  
    pool1 = pooling_unit(conv1_1, stage=model_depth-3, pooling=pooling)

    conv2_1 = standard_unit(pool1, stage=model_depth-2, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)
    conv2_2 = layers.UpSampling3D(size=2**1)(conv2_1)
    conv2_2 = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='out_2', kernel_initializer='he_normal', padding='same')(conv2_2)  
    pool2 = pooling_unit(conv2_1, stage=model_depth-2, pooling=pooling)

    conv3_1 = standard_unit(pool2, stage=model_depth-1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)
    conv3_2 = layers.UpSampling3D(size=2**2)(conv3_1)
    conv3_2 = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='out_3', kernel_initializer='he_normal', padding='same')(conv3_2)     
    pool3 = pooling_unit(conv3_1, stage=model_depth-1, pooling=pooling)

    conv4_1 = standard_unit(pool3, stage=model_depth, nb_filter=nb_filter[model_depth], layer_act=layer_act)
    conv4_2 = layers.UpSampling3D(size=2**3)(conv4_1)
    conv4_2 = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='out_4', kernel_initializer='he_normal', padding='same')(conv4_2)   
    
    conv3_2 = up_unit(conv4_1, conv3_1, stage=model_depth-1, nb_filter=nb_filter[model_depth-1], layer_act=layer_act)  
    conv3_3 = layers.UpSampling3D(size=2**2,)(conv3_2)
    conv3_3 = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='out_5', kernel_initializer='he_normal', padding='same')(conv3_3)                           
    conv2_2 = up_unit(conv3_2, conv2_1, stage=model_depth-2, nb_filter=nb_filter[model_depth-2], layer_act=layer_act)   
    conv2_3 = layers.UpSampling3D(size=2**1)(conv2_2)
    conv2_3 = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='out_6', kernel_initializer='he_normal', padding='same')(conv2_3)                          
    conv1_2 = up_unit(conv2_2, conv1_1, stage=model_depth-3, nb_filter=nb_filter[model_depth-3], layer_act=layer_act)
    conv1_3 = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='out_7', kernel_initializer='he_normal', padding='same')(conv1_2)
    
    
    if first==1 and end==1:
        return Model(inputs=img_input, outputs=[conv1_2, last_out])
    elif first==2 and end==2:
        return Model(inputs=img_input, outputs=[conv1_2, conv2_2, conv2_4, conv1_3])    
    elif first==3 and end==3:
        return Model(inputs=img_input, outputs=[conv1_2, conv2_2, conv3_2, conv3_3, conv2_3, conv1_3])
    elif first==4 and end==3:
        return Model(inputs=img_input, outputs=[conv1_2, conv2_2, conv3_2, conv4_1, conv3_3, conv2_3, conv1_3])
    elif first==0 and end==1:
        return Model(inputs=img_input, outputs=[last_out])
    elif first==0 and end==2:
        return Model(inputs=img_input, outputs=[conv2_4, last_out])
    elif first==0 and end==3:
        return Model(inputs=img_input, outputs=[conv3_3, conv2_4, last_out])



