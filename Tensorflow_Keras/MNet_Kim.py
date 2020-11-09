from keras.layers import *

from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras import backend as K
from keras.models import Model

K.set_image_data_format('channels_last')

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)

# Tensorflow to Keras
# Fully Automatic Segmentation of Coronary Arteries Based on Deep Neural Network in Intravascular Ultrasound Images

def MNet_Kim_model(input_shape=None, classes=3):

    
    # Encoder
    img_input = Input(shape = input_shape)
    
    max_1 = MaxPooling2D((2,2))(img_input) # 128
    max_2 = MaxPooling2D((2,2))(max_1) # 64
    max_3 = MaxPooling2D((2,2))(max_2) # 32
    max_4 = MaxPooling2D((2,2))(max_3) # 16
    
    x = Conv2D(16, (3, 3), activation = None, strides = 1, padding = 'same')(img_input) # 256
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x1_cat = concatenate([x, img_input])
    x = Conv2D(32, (3, 3), activation = None, strides = 1, padding = 'same')(x1_cat) 
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)  
    x = MaxPooling2D((2,2))(x1)
    
     
    x2_cat = concatenate([x, max_1]) # 128
    x = Conv2D(32, (3, 3), activation = None, strides = 1, padding = 'same')(x2_cat) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x2_cat2 = concatenate([x, x2_cat])
    x = Conv2D(48, (3, 3), activation = None, strides = 1, padding = 'same')(x2_cat2) 
    x = BatchNormalization()(x)
    x2 = Activation('relu')(x)     
    x = MaxPooling2D((2,2))(x2)
    
    
    x3_cat = concatenate([x, max_2]) # 64
    x = Conv2D(48, (3, 3), activation = None, strides = 1, padding = 'same')(x3_cat) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x3_cat2 = concatenate([x, x3_cat])
    x = Conv2D(64, (3, 3), activation = None, strides = 1, padding = 'same')(x3_cat2) 
    x = BatchNormalization()(x)
    x3 = Activation('relu')(x)     
    x = MaxPooling2D((2,2))(x3)
    
 
    x4_cat = concatenate([x, max_3]) # 32
    x = Conv2D(64, (3, 3), activation = None, strides = 1, padding = 'same')(x4_cat) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x4_cat2 = concatenate([x, x4_cat])
    x = Conv2D(80, (3, 3), activation = None, strides = 1, padding = 'same')(x4_cat2) 
    x = BatchNormalization()(x)
    x4 = Activation('relu')(x)     
    x = MaxPooling2D((2,2))(x4)


    x5_cat = concatenate([x, max_4]) # 16
    x = Conv2D(80, (3, 3), activation = None, strides = 1, padding = 'same')(x5_cat) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x5_cat2 = concatenate([x, x5_cat])
    x = Conv2D(160, (3, 3), activation = None, strides = 1, padding = 'same')(x5_cat2) 
    x = BatchNormalization()(x)
    x5 = Activation('relu')(x)     
    x = Conv2D(80, (3, 3), activation = None, strides = 1, padding = 'same')(x5) 
    x = BatchNormalization()(x)
    x_bridge = Activation('relu')(x)  
    
    # Decoder
    up4 = UpSampling2D((2,2))(x_bridge) # 32 
    x = BatchNormalization()(up4)
    x = Activation('relu')(x)    
    up4_cat = concatenate([x, x4, max_3])
    x = Conv2D(80, (3, 3), activation = None, strides = 1, padding = 'same')(up4_cat) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)  
    up4_cat2 = concatenate([x, up4_cat])
    x = Conv2D(64, (3, 3), activation = None, strides = 1, padding = 'same')(up4_cat2) 
    x = BatchNormalization()(x)
    x4_out = Activation('relu')(x)
    
    up3 = UpSampling2D((2,2))(x4_out)  # 64
    x = BatchNormalization()(up3)
    x = Activation('relu')(x)    
    up3_cat = concatenate([x, x3, max_2])
    x = Conv2D(64, (3, 3), activation = None, strides = 1, padding = 'same')(up3_cat) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)  
    up3_cat2 = concatenate([x, up3_cat])
    x = Conv2D(48, (3, 3), activation = None, strides = 1, padding = 'same')(up3_cat2) 
    x = BatchNormalization()(x)
    x3_out = Activation('relu')(x)
    
    up2 = UpSampling2D((2,2))(x3_out)  # 128
    x = BatchNormalization()(up2)
    x = Activation('relu')(x)    
    up2_cat = concatenate([x, x2, max_1])
    x = Conv2D(48, (3, 3), activation = None, strides = 1, padding = 'same')(up2_cat) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)  
    up2_cat2 = concatenate([x, up2_cat])
    x = Conv2D(32, (3, 3), activation = None, strides = 1, padding = 'same')(up2_cat2) 
    x = BatchNormalization()(x)
    x2_out = Activation('relu')(x)
    
    
    up1 = UpSampling2D((2,2))(x2_out)  # 256
    x = BatchNormalization()(up1)
    x = Activation('relu')(x)    
    up1_cat = concatenate([x, x1, img_input])
    x = Conv2D(32, (3, 3), activation = None, strides = 1, padding = 'same')(up1_cat) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)  
    up1_cat2 = concatenate([x, up1_cat])
    x = Conv2D(16, (3, 3), activation = None, strides = 1, padding = 'same')(up1_cat2) 
    x = BatchNormalization()(x)
    x1_out = Activation('relu')(x)    
    
    
    # Multi-label loss function layer
    x5_out_cat = concatenate([max_4, x_bridge])
    x5_out_up1 = UpSampling2D((2,2))(x5_out_cat)  # 32
    x5_out_up1 = BatchNormalization()(x5_out_up1)
    x5_out_up1 = Activation('relu')(x5_out_up1)
    
    x5_out_up2 = UpSampling2D((2,2))(x_bridge)  # 32
    x5_out_up2 = BatchNormalization()(x5_out_up2)
    x5_out_up2 = Activation('relu')(x5_out_up2) 
    x5_out_up_cat = concatenate([x4_out, x5_out_up1, x5_out_up2])
    
    x5_final = UpSampling2D((8,8))(x5_out_up_cat)
    x5_final = Conv2D(classes, (3, 3), activation = 'softmax', strides = 1, padding = 'same', name = 'x5_final')(x5_final)
    
    x4_out_cat = concatenate([max_3, x4_out])
    x4_out_up1 = UpSampling2D((2,2))(x4_out_cat)  # 64
    x4_out_up1 = BatchNormalization()(x4_out_up1)
    x4_out_up1 = Activation('relu')(x4_out_up1) 
    
    x4_out_up2 = UpSampling2D((2,2))(x5_out_up_cat)  # 64
    x4_out_up2 = BatchNormalization()(x4_out_up2)
    x4_out_up2 = Activation('relu')(x4_out_up2) 
    x4_out_up_cat = concatenate([x3_out, x4_out_up1, x4_out_up2])
    
    x4_final = UpSampling2D((4,4))(x4_out_up_cat)
    x4_final = Conv2D(classes, (3, 3), activation = 'softmax', strides = 1, padding = 'same', name = 'x4_final')(x4_final)  
    
    x3_out_cat = concatenate([max_2, x3_out])
    x3_out_up1 = UpSampling2D((2,2))(x3_out_cat)  # 128
    x3_out_up1 = BatchNormalization()(x3_out_up1)
    x3_out_up1 = Activation('relu')(x3_out_up1) 
    
    x3_out_up2 = UpSampling2D((2,2))(x4_out_up_cat)  # 128
    x3_out_up2 = BatchNormalization()(x3_out_up2)
    x3_out_up2 = Activation('relu')(x3_out_up2) 
    x3_out_up_cat = concatenate([x2_out, x3_out_up1, x3_out_up2])
    
    x3_final = UpSampling2D((2,2))(x3_out_up_cat)
    x3_final = Conv2D(classes, (3, 3), activation = 'softmax', strides = 1, padding = 'same', name = 'x3_final')(x3_final)     
    
    x2_out_cat = concatenate([max_1, x2_out])
    x2_out_up1 = UpSampling2D((2,2))(x2_out_cat)  #  256
    x2_out_up1 = BatchNormalization()(x2_out_up1)
    x2_out_up1 = Activation('relu')(x2_out_up1) 
    
    x2_out_up2 = UpSampling2D((2,2))(x3_out_up_cat)  # 256
    x2_out_up2 = BatchNormalization()(x2_out_up2)
    x2_out_up2 = Activation('relu')(x2_out_up2) 
    x2_out_up_cat = concatenate([x1_out, x2_out_up1, x2_out_up2])
    
    x1_final = Conv2D(64, (3, 3), activation = None, strides = 1, padding = 'same', name = 'x1_final_conv')(x2_out_up_cat)    
    x1_final = BatchNormalization()(x1_final)
    x1_final = Activation('relu')(x1_final) 
    x1_final = Conv2D(classes, (3, 3), activation = 'softmax', strides = 1, padding = 'same', name = 'x1_final')(x1_final)     

    #model = Model(img_input, [x5_final, x4_final, x3_final, x1_final])
    model = Model(img_input, [x1_final, x3_final, x4_final, x5_final])

    return model

