from keras.layers import MaxPooling2D, Dropout, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, concatenate, Input
from keras.models import Model
from keras import backend as K

def block_1(input_layer, nb_filter):
    conv1 = Conv2D(nb_filter, (3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(nb_filter, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), )(conv1)
    return conv1, pool1

def block_2(input_layer, nb_filter):
    conv2 = Conv2D(nb_filter, (3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(nb_filter, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), )(conv2)
    return conv2, pool2

def block_concat(input_layer, concat_layer, nb_filter, bn_axis = 3):
    up1_2 = Conv2DTranspose(nb_filter, (2, 2), strides=(2, 2), padding='same')(input_layer) 
    conv1_2 = concatenate([up1_2, concat_layer], axis=bn_axis)
    conv1_2 = Conv2D(nb_filter, (3, 3), activation='relu', padding='same')(conv1_2)
    conv1_2 = Conv2D(nb_filter, (3, 3), activation='relu', padding='same')(conv1_2) 
    return conv1_2

def unetpp(num_class=1, input_shape=(256,256,1), deep_supervision=4, nb_filter = [32,64,128,256,512]):
    assert len(nb_filter)-1 == deep_supervision, 'Match the same depth of block'
    assert deep_supervision not in [1,2,3,4], 'deep_supervision should be between 1 and 4'
    
    img_input = Input(shape=input_shape)

    conv1_1, pool1 = block_1(img_input, nb_filter[0])    
    conv2_1, pool2 = block_2(pool1, nb_filter[1])
    conv1_2 = block_concat(conv2_1, conv1_1, nb_filter[0])

    conv3_1, pool3 = block_1(pool2, nb_filter[2])
    conv2_2 = block_concat(conv3_1, conv2_1, nb_filter[1])
    conv1_3 = block_concat(conv2_2, conv1_1, nb_filter[0])

    conv4_1, pool4 = block_1(pool3, nb_filter[3])
    conv3_2 = block_concat(conv4_1, conv3_1, nb_filter[2])
    conv2_3 = block_concat(conv3_2, conv2_1, nb_filter[1])
    conv1_4 = block_concat(conv2_3, conv1_1, nb_filter[0])
    
    # branch
    conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(pool4)
    conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(conv5_1)

    conv4_2 = block_concat(conv5_1, nb_filter[3])
    conv3_3 = block_concat(conv4_2, nb_filter[2])
    conv2_4 = block_concat(conv3_3, nb_filter[1])
    conv1_5 = block_concat(conv2_4, nb_filter[0])

    # output
    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision==4:
        return Model(input=img_input, output=[nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4])
    elif deep_supervision==3:
        return Model(input=img_input, output=[nestnet_output_2, nestnet_output_3, nestnet_output_4])
    elif deep_supervision==2:
        return Model(input=img_input, output=[nestnet_output_3, nestnet_output_4])
    else:
        return Model(input=img_input, output=[nestnet_output_4])  
    
def unetppWedge(num_shape=1, num_edge=1, input_shape=(256,256,1), deep_supervision=4, nb_filter = [32,64,128,256,512]):
    assert len(nb_filter)-1 == deep_supervision, 'Match the same depth of block'
    assert deep_supervision in [1,2,3,4], 'deep_supervision should be between 1 and 4'
    
    img_input = Input(shape=input_shape)

    conv1_1, pool1 = block_1(img_input, nb_filter[0])    
    conv2_1, pool2 = block_2(pool1, nb_filter[1])
    conv1_2 = block_concat(conv2_1, conv1_1, nb_filter[0])
    conv1_2_e = block_concat(conv2_1, conv1_1, nb_filter[0])

    conv3_1, pool3 = block_1(pool2, nb_filter[2])
    conv2_2 = block_concat(conv3_1, conv2_1, nb_filter[1])
    conv1_3 = block_concat(conv2_2, conv1_1, nb_filter[0])
    conv2_2_e = block_concat(conv3_1, conv2_1, nb_filter[1])
    conv1_3_e = block_concat(conv2_2, conv1_1, nb_filter[0])

    conv4_1, pool4 = block_1(pool3, nb_filter[3])
    conv3_2 = block_concat(conv4_1, conv3_1, nb_filter[2])
    conv2_3 = block_concat(conv3_2, conv2_1, nb_filter[1])
    conv1_4 = block_concat(conv2_3, conv1_1, nb_filter[0])
    conv3_2_e = block_concat(conv4_1, conv3_1, nb_filter[2])
    conv2_3_e = block_concat(conv3_2, conv2_1, nb_filter[1])
    conv1_4_e = block_concat(conv2_3, conv1_1, nb_filter[0])
    
    # branch
    conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(pool4)
    conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(conv5_1)

    conv4_2 = block_concat(conv5_1, nb_filter[3])
    conv3_3 = block_concat(conv4_2, nb_filter[2])
    conv2_4 = block_concat(conv3_3, nb_filter[1])
    conv1_5 = block_concat(conv2_4, nb_filter[0])
    
    # branch
    conv5_1_e = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(pool4)
    conv5_1_e = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(conv5_1_e)

    conv4_2_e = block_concat(conv5_1_e, nb_filter[3])
    conv3_3_e = block_concat(conv4_2_e, nb_filter[2])
    conv2_4_e = block_concat(conv3_3_e, nb_filter[1])
    conv1_5_e = block_concat(conv2_4_e, nb_filter[0])

    nestnet_output_1 = Conv2D(num_shape, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_shape, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_shape, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_shape, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    nestnet_output_1_e = Conv2D(num_edge, (1, 1), activation='sigmoid', name='output_1_e', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2_e)
    nestnet_output_2_e = Conv2D(num_edge, (1, 1), activation='sigmoid', name='output_2_e', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3_e)
    nestnet_output_3_e = Conv2D(num_edge, (1, 1), activation='sigmoid', name='output_3_e', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4_e)
    nestnet_output_4_e = Conv2D(num_edge, (1, 1), activation='sigmoid', name='output_4_e', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5_e)

    if deep_supervision==4:
        return Model(input=img_input, output=[nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4, nestnet_output_1_e, nestnet_output_2_e, nestnet_output_3_e, nestnet_output_4_e])
    elif deep_supervision==3:
        return Model(input=img_input, output=[nestnet_output_2, nestnet_output_3, nestnet_output_4, nestnet_output_2_e, nestnet_output_3_e, nestnet_output_4_e])
    elif deep_supervision==2:
        return Model(input=img_input, output=[nestnet_output_3, nestnet_output_4, nestnet_output_3_e, nestnet_output_4_e])
    else:
        return Model(input=img_input, output=[nestnet_output_4, nestnet_output_4_e]) 