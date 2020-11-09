# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:04:00 2020

@author: JEpark
"""
import tensorflow as tf
if tf.__version__=='1.15.0':
    from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Input, Conv2DTranspose
    from tensorflow.keras.layers import core, concatenate, LayerNormalization
    from tensorflow.keras.models import Model
#     from keras_layer_normalization import LayerNormalization
else:    
    from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Input, Conv2DTranspose
    from keras.layers import core, concatenate
    from keras.models import Model
    from keras_layer_normalization import LayerNormalization

def conv_block(x, filters, kernel_size=(3,3), padding='same', mode_norm='batch', data_format='channels_last'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, padding=padding, kernel_initializer='he_normal')(x)
    if mode_norm=='batch':
        x = BatchNormalization()(x)
    else :
        x = LayerNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def deconv_block(x, skips, filters, kernel_size=(3,3), strides=(2,2), mode_norm='batch', padding='same', data_format='channels_last'):
    x = Conv2DTranspose(filters=filters, kernel_size=(2,2), strides=strides, padding=padding)(x)
    x = concatenate([skips, x], axis=3)    
    x = conv_block(x, filters=filters, mode_norm=mode_norm)
    x = conv_block(x, filters=filters, mode_norm=mode_norm)
    
    return x

#####################################################################################################################################
# Unet w/ Edge w/ multi-input w/ supervision (YS)
def unet_edge_sv(img_shape, n_shapes, n_edges, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(img_shape)
    x = inputs
    # multi-inputs
    avg = inputs
    avgs = []
    for i in range(depth):
        avg = AveragePooling2D(pool_size=(2, 2))(avg)
        avgs.append(avg)
    ## encoder
    skips = []
    for i in range(depth):                
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        avg2 = conv_block(avgs[i], features, mode_norm=mode_norm, data_format=data_format)
        x = concatenate([x, avg2])

        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    
    ## decoder
    outs = []
    # for shape
    x1 = x
    filters = features
    up_size = depth * 2
    for i in reversed(range(depth)):
        filters = filters // 2
        x1 = deconv_block(x1, skips[i], filters, mode_norm=mode_norm, data_format=data_format)
    
        out_shape = UpSampling2D(size=(up_size, up_size), name='before_out_shape_{}'.format(i))(x1)
        outs.append(Conv2D(n_shapes, (3,3), activation='softmax', padding='same', name='out_shape_{}'.format(i))(out_shape))
        up_size = up_size // 2
    # for edge 
    x2 = x
    filters = features
    up_size = depth * 2
    for i in reversed(range(depth)):
        filters = filters // 2
        x2 = deconv_block(x2, skips[i], filters, mode_norm=mode_norm, data_format=data_format)
    
        out_edge = UpSampling2D(size=(up_size, up_size), name='before_out_edge_{}'.format(i))(x2)
        outs.append(Conv2D(n_edges, (3,3), activation='softmax', padding='same', name='out_edge_{}'.format(i))(out_edge))
        up_size = up_size // 2
    
    model = Model(inputs=inputs, outputs=outs)
    
    model.summary()
    return model

#####################################################################################################################################

#U-Net
def unet(img_shape, n_shapes, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(img_shape)
    x = inputs    

    skips = []
    for i in range(depth):
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = deconv_block(x, skips[i], features, data_format=data_format)

    conv6 = Conv2D(n_shapes, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    
    model.summary()
    return model

# Unet w/ Edge
def unet_edge(img_shape, n_shapes, n_edges, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(img_shape)
    x = inputs
    skips = []
    for i in range(depth):
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)

        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)

    x1 = x
    for i in reversed(range(depth)):
        features = features // 2
        x1 = deconv_block(x1, skips[i], features, data_format=data_format)
    
    conv6 = Conv2D(n_shapes, (1, 1), padding='same', data_format=data_format)(x1)
    conv7 = core.Activation('sigmoid')(conv6)
    x2 = x
    for i in reversed(range(depth)):
        features = features // 2
        x2 = deconv_block(x2, skips[i], features, data_format=data_format)
        
    conv8 = Conv2D(n_edges, (1, 1), padding='same', data_format=data_format)(x2)
    conv9 = core.Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=[conv7, conv9])
    
    model.summary()
    return model

# Unet w/ Edge w/ multi-inputs w/o deep-supervision
def unet_edge_avg(img_shape, n_shapes, n_edges, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(img_shape)
    x = inputs

    avg = inputs
    avgs = []
    for i in range(depth):
        avg = AveragePooling2D(pool_size=(2, 2))(avg)
        avgs.append(avg)

    skips = []
    for i in range(depth):
        
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        avg2 = conv_block(avgs[i], features, mode_norm='batch', data_format=data_format)
        x = concatenate([x, avg2])

        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)

    x1 = x
    for i in reversed(range(depth)):
        features = features // 2
        x1 = deconv_block(x1, skips[i], features, mode_norm=mode_norm, data_format=data_format)
    
    conv6 = Conv2D(n_shapes, (1, 1), padding='same', data_format=data_format)(x1)
    conv7 = core.Activation('sigmoid')(conv6)
    x2 = x
    for i in reversed(range(depth)):
        features = features // 2
        x2 = deconv_block(x2, skips[i], features, data_format=data_format)
        
    conv8 = Conv2D(n_edges, (1, 1), padding='same', data_format=data_format)(x2)
    conv9 = core.Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=[conv7, conv9])
    
    model.summary()
    return model


# Unet w/o Edge w/ supervision 
def unet_sv(img_shape, n_shapes, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(img_shape)
    x = inputs

    avg = inputs
    avgs = []
    for i in range(depth):
        avg = AveragePooling2D(pool_size=(2, 2))(avg)
        avgs.append(avg)

    skips = []
    for i in range(depth):
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        avg2 = conv_block(avgs[i], features, mode_norm=mode_norm, data_format=data_format)
        x = concatenate([x, avg2])

        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)

    outs = []
    up_size = depth * 2
    for i in reversed(range(depth)):
        features = features // 2
        x = deconv_block(x, skips[i], features, mode_norm=mode_norm, data_format=data_format)
    
        conv_out = UpSampling2D(size=(up_size, up_size))(x)
        outs.append(Conv2D(n_shapes, (3,3), activation='softmax', padding='same', name='out_{}'.format(i))(conv_out))
        up_size = up_size // 2
        
    model = Model(inputs=inputs, outputs=outs)
    
    model.summary()
    return model


# Unet w/o Edge w/ supervision 
def unet_sv_outSz(in_shape, out_shape, n_shapes, depth = 4, features = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(in_shape)
    size = out_shape // int(in_shape[0])
    print(size)
    
    x = inputs

    avg = inputs
    avgs = []
    for i in range(depth):
        avg = AveragePooling2D(pool_size=(2, 2))(avg)
        avgs.append(avg)

    skips = []
    for i in range(depth):        
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        avg2 = conv_block(avgs[i], features, mode_norm=mode_norm, data_format=data_format)
        x = concatenate([x, avg2])
        
        features = features * 2
        
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, features, mode_norm=mode_norm, data_format=data_format)

    outs = []
    up_size = depth * (2 * size)
    for i in reversed(range(depth)):
        features = features // 2
        x = deconv_block(x, skips[i], features, mode_norm=mode_norm, data_format=data_format)    
        conv_out = UpSampling2D(size=(up_size, up_size))(x)
        outs.append(Conv2D(n_shapes, 3, activation='softmax', padding='same')(conv_out))
        up_size = up_size // 2
                
    model = Model(inputs=inputs, outputs=outs)
    
    model.summary()
    return model

# Unet w/ Edge w/ supervision (YS)
def unet_edge_sv_outSz(in_shape, out_shape, n_shapes, n_edges, depth = 4, filters = 32, mode_norm='batch', data_format='channels_last'):
    inputs = Input(in_shape)
    size = out_shape // int(in_shape[0])
    print(size)
    
    x = inputs

    avg = inputs
    avgs = []
    for i in range(depth):
        avg = AveragePooling2D(pool_size=(2, 2))(avg)
        avgs.append(avg)

    skips = []
    for i in range(depth):        
        x = conv_block(x, filters=filters, mode_norm=mode_norm, data_format=data_format)
        x = conv_block(x, filters=filters, mode_norm=mode_norm, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        avg2 = conv_block(avgs[i], filters=filters, mode_norm=mode_norm, data_format=data_format)
        x = concatenate([x, avg2])
        
        filters = filters * 2
        
    x = conv_block(x, filters=filters, mode_norm=mode_norm, data_format=data_format)
    x = conv_block(x, filters=filters, mode_norm=mode_norm, data_format=data_format)

    x1 = x
    outs = []
    up_size = depth * 2
    for i in reversed(range(depth)):
        filters = filters // 2
        x1 = deconv_block(x1, skips[i], filters=filters, mode_norm=mode_norm, data_format=data_format)
        
        conv_out = UpSampling2D(size=(up_size, up_size))(x1)
        conv_out = Conv2D(n_shapes, 3, activation='softmax', padding='same')(conv_out)
        outs.append(conv_out)
        up_size = up_size // 2
        
    x2 = x
    up_size = depth * 2
    for i in reversed(range(depth)):
        filters = filters // 2
        x2 = deconv_block(x2, skips[i], filters=filters, mode_norm=mode_norm, data_format=data_format)
        
        conv_out = UpSampling2D(size=(up_size, up_size))(x2)
        conv_out = Conv2D(n_edges, 3, activation='softmax', padding='same')(conv_out)
        outs.append(conv_out)
        up_size = up_size // 2
        
    model = Model(inputs=inputs, outputs=outs)
    
    model.summary()
    return model