import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
def vgg19(x, filters):
    x1 = Conv3D(filters, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x)
    x1 = Conv3D(filters, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x1)
    x1 = BatchNormalization()(x1)
    x2 = MaxPooling3D(strides=2)(x1)
    
    x2 = Conv3D(filters*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x2)
    x2 = Conv3D(filters*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x2)
    x2 = BatchNormalization()(x2)
    x3 = MaxPooling3D(strides=2)(x2)
    
    x3 = Conv3D(filters*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x3)
    x3 = Conv3D(filters*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x3)
    x3 = Conv3D(filters*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x3)
    x3 = Conv3D(filters*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x3)
    x3 = BatchNormalization()(x3)
    x4 = MaxPooling3D(strides=2)(x3)
    
    x4 = Conv3D(filters*2*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x4)
    x4 = Conv3D(filters*2*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x4)
    x4 = Conv3D(filters*2*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x4)
    x4 = Conv3D(filters*2*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x4)
    x4 = BatchNormalization()(x4)
    x5 = MaxPooling3D(strides=2)(x4)
    
    x5 = Conv3D(filters*2*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x5)
    x5 = Conv3D(filters*2*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x5)
    x5 = Conv3D(filters*2*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x5)
    x5 = Conv3D(filters*2*2*2, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x5)
    x5 = BatchNormalization()(x5)
    x6 = MaxPooling3D(strides=2)(x5)
    return x1, x2, x3, x4, x5, x6

def decoder(x1, x2, factor, filters, activation):
    x = concatenate([UpSampling3D(factor)(x2), x1])
    x = Conv3D(filters, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x)
    x = Conv3D(filters, kernel_size=3, kernel_initializer = 'he_normal', padding="same")(x)
    x = BatchNormalization()(x)
    return x
    
    
def VggUnet(img_size, num_classes, filters=16, activation='relu'):
    inputs = Input(shape=img_size)

    previous_block_activation = inputs  # Set aside residual

    x1, x2, x3, x4, x5, x6 = vgg19(inputs, filters)
    x = decoder(x5, x6, 2, filters*2*2*2, activation)
    x = decoder(x4, x, 2, filters*2*2, activation)
    x = decoder(x3, x, 2, filters*2, activation)
    x = decoder(x2, x, 2, filters, activation)
    x = decoder(x1, x, 2, 1, activation)
    # Add a per-pixel classification layer
    outputs = Conv3D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = Model(inputs, outputs)
    return model
