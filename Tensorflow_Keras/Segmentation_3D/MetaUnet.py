import tensorflow as tf
from tensorflow.keras import Input, Model, layers, utils


def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    x = layers.Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), 
                      activation=None, name='conv'+stage+'_1', 
                      kernel_initializer = 'he_normal', padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size),
                      activation=None, name='conv'+stage+'_2',
                      kernel_initializer = 'he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def unet(shape, num_class=1, filters=32, activation='sigmoid'):    
    nb_filter = [filters*2**i for i in range(5)]
    bn_axis = -1
    img_input = Input(shape=shape, name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])

    up3_3 = layers.Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up33', padding='same')(conv4_1)
    conv3_3 = layers.concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = layers.Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = layers.concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = layers.Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = layers.concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='output', 
                                kernel_initializer = 'he_normal', padding='same')(conv1_5)
    
    return Model(inputs=img_input, outputs=unet_output[...,shape[3]//2,:]) # tf.reduce_sum(unet_output,axis=3)

# with multiple branches
def Unet3D_2b(shape, num_class=1, filters=32, activation='sigmoid'):    
    nb_filter = [filters*2**i for i in range(5)]
    bn_axis = -1
    img_input = Input(shape=shape, name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    
    ################################################################################################################# branch 1
    up3_3 = layers.Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up33', padding='same')(conv4_1)
    conv3_3 = layers.concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = layers.Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = layers.concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = layers.Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = layers.concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])
    
    unet_output = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='output', 
                                kernel_initializer = 'he_normal', padding='same')(conv1_5)
    ################################################################################################################# branch 2
    up3_3b = layers.Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up33b', padding='same')(conv4_1)
    conv3_3b = layers.concatenate([up3_3b, conv3_1], name='merge33', axis=bn_axis)
    conv3_3b = standard_unit(conv3_3b, stage='33', nb_filter=nb_filter[2])

    up2_4b = layers.Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up24b', padding='same')(conv3_3b)
    conv2_4b = layers.concatenate([up2_4b, conv2_1], name='merge24', axis=bn_axis)
    conv2_4b = standard_unit(conv2_4b, stage='24', nb_filter=nb_filter[1])

    up1_5b = layers.Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up15b', padding='same')(conv2_4b)
    conv1_5b = layers.concatenate([up1_5b, conv1_1], name='merge15', axis=bn_axis)
    conv1_5b = standard_unit(conv1_5b, stage='15', nb_filter=nb_filter[0])
    
    unet_outputb = layers.Conv3D(num_class, (1, 1, 1), activation=activation, name='outputb', 
                                kernel_initializer = 'he_normal', padding='same')(conv1_5b)
    
    return Model(inputs=img_input, outputs=[unet_output,unet_outputb])