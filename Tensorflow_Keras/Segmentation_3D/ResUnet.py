import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.regularizers import l2

def ResUnet_re(shape, base_filter=32, layer_activation='relu', padding='same', depth=4, num_classes=1):
    if num_classes == 1:
        activation='sigmoid'
    else:
        activation='softmax'
        
    inputs = Input((shape[0], shape[1], shape[2], 1))
    filters = base_filter
    conv, conc, pool, up = [], [], [inputs], []
    
    for d in range(depth):
        temp = Conv3D(filters, (3, 3, 3), activation=layer_activation, padding=padding)(pool[-1])
        conv.append(Conv3D(filters, (3, 3, 3), activation=layer_activation, padding=padding)(temp))
        conc.append(concatenate([pool[-1], conv[-1]], axis=4))
        pool.append(MaxPooling3D(pool_size=(2, 2, 2))(conc[-1]))
        filters *= 2

    conv.append(Conv3D(filters, (3, 3, 3), activation=layer_activation, padding=padding)(pool[-1]))
    conv.append(Conv3D(filters, (3, 3, 3), activation=layer_activation, padding=padding)(conv[-1]))
    conc.append(concatenate([pool[-1], conv[-1]], axis=4))
    
    for d in reversed(range(depth)):
        filters /= 2
        up.append(concatenate([Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc[-1]), conv[d]], axis=-1))
        temp = Conv3D(filters, (3, 3, 3), activation=layer_activation, padding=padding)(up[-1])
        conv.append(Conv3D(filters, (3, 3, 3), activation=layer_activation, padding=padding)(temp))
        conc.append(concatenate([up[-1], conv[-1]], axis=4))

    outs = Conv3D(num_classes, (1, 1, 1), activation=activation)(conc[-1])

    model = Model(inputs=[inputs], outputs=[outs])
    return model

def ResUnet(shape, base_filter=32, layer_activation='relu', padding='same', activation='sigmoid'):
    inputs = Input((shape[0], shape[1], shape[2], 1))

    conv1 = Conv3D(base_filter*2, (3, 3, 3), activation=layer_activation, padding=padding)(inputs)
    conv1 = Conv3D(base_filter*2, (3, 3, 3), activation=layer_activation, padding=padding)(conv1)
    conc1 = concatenate([pool1, conv2], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc)
    
    conv2 = Conv3D(base_filter*2, (3, 3, 3), activation=layer_activation, padding=padding)(pool1)
    conv2 = Conv3D(base_filter*2, (3, 3, 3), activation=layer_activation, padding=padding)(conv2)
    conc2 = concatenate([pool1, conv2], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc2)

    conv3 = Conv3D(base_filter*2*2, (3, 3, 3), activation=layer_activation, padding=padding)(pool2)
    conv3 = Conv3D(base_filter*2*2, (3, 3, 3), activation=layer_activation, padding=padding)(conv3)
    conc3 = concatenate([pool2, conv3], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc3)

    conv4 = Conv3D(base_filter*2*2*2, (3, 3, 3), activation=layer_activation, padding=padding)(pool3)
    conv4 = Conv3D(base_filter*2*2*2, (3, 3, 3), activation=layer_activation, padding=padding)(conv4)
    conc4 = concatenate([pool3, conv4], axis=4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc4)

    conv5 = Conv3D(base_filter*2*2*2*2, (3, 3, 3), activation=layer_activation, padding=padding)(pool4)
    conv5 = Conv3D(base_filter*2*2*2*2, (3, 3, 3), activation=layer_activation, padding=padding)(conv5)
    conc5 = concatenate([pool4, conv5], axis=4)

    up6 = concatenate([Conv3DTranspose(base_filter*2*2*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc5), conv4], axis=4)
    conv6 = Conv3D(base_filter*2*2*2, (3, 3, 3), activation=layer_activation, padding=padding)(up6)
    conv6 = Conv3D(base_filter*2*2*2, (3, 3, 3), activation=layer_activation, padding=padding)(conv6)
    conc6 = concatenate([up6, conv6], axis=4)

    up7 = concatenate([Conv3DTranspose(base_filter*2*2, (2, 2, 2), strides=(2, 2, 2), padding=padding)(conc6), conv3], axis=4)
    conv7 = Conv3D(base_filter*2*2, (3, 3, 3), activation=layer_activation, padding=padding)(up7)
    conv7 = Conv3D(base_filter*2*2, (3, 3, 3), activation=layer_activation, padding=padding)(conv7)
    conc7 = concatenate([up7, conv7], axis=4)

    up8 = concatenate([Conv3DTranspose(base_filter*2, (2, 2, 2), strides=(2, 2, 2), padding=padding)(conc7), conv2], axis=4)
    conv8 = Conv3D(base_filter*2, (3, 3, 3), activation=layer_activation, padding=padding)(up8)
    conv8 = Conv3D(base_filter*2, (3, 3, 3), activation=layer_activation, padding=padding)(conv8)
    conc8 = concatenate([up8, conv8], axis=4)

    up9 = concatenate([Conv3DTranspose(base_filter, (2, 2, 2), strides=(2, 2, 2), padding=padding)(conc8), conv1], axis=4)
    conv9 = Conv3D(base_filter, (3, 3, 3), activation=layer_activation, padding=padding)(up9)
    conv9 = Conv3D(base_filter, (3, 3, 3), activation=layer_activation, padding=padding)(conv9)
    conc9 = concatenate([up9, conv9], axis=4)

    conv10 = Conv3D(1, (1, 1, 1), activation=activation)(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model