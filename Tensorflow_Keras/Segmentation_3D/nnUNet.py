import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv3D,Conv3DTranspose, concatenate, BatchNormalization, LeakyReLU, ZeroPadding3D
from tensorflow.keras import Model, Input


def nnUNet(input_shape, feature_maps=16, max_fa=480, num_pool=6, k_init='he_normal', num_classes=1, activation="softmax"):
    """Create nnU-Net. This implementations tries to be a Keras version of the original nnU-Net presented in
       `nnU-Net Github <https://github.com/MIC-DKFZ/nnUNet>`_.
                                                                                
       Parameters
       ----------
       image_shape : tuple
           Dimensions of the input image.              
                                                                                
       feature_map : ints, optional
           Feature maps to start with in the first level of the U-Net (will be duplicated on each level). 

       max_fa : int, optional
           Number of maximum feature maps allowed to used in conv layers.
        
       num_pool : int, optional
           number of pooling (downsampling) operations to do.

       k_init : string, optional
           Kernel initialization for convolutional layers.                                                         
                                                                           
       n_classes: int, optional
           Number of classes.
                                                                           
       Returns
       -------                                                                 
       model : Keras model
           Model containing the U-Net.              
    """
                          
    x = Input(input_shape)                                                      
    #x = Input(image_shape)                                                     
    inputs = x
        
    l=[]
    seg_outputs = []
    fa_save = []
    fa = feature_maps

    # ENCODER
    x = StackedConvLayers(x, fa, k_init, first_conv_stride=1)
    fa_save.append(fa)
    fa = fa*2 if fa*2 < max_fa else max_fa
    l.append(x)

    # conv_blocks_context
    for i in range(num_pool-1):
        x = StackedConvLayers(x, fa, k_init)
        fa_save.append(fa)
        fa = fa*2 if fa*2 < max_fa else max_fa
        l.append(x)

    # BOTTLENECK
    x = StackedConvLayers(x, fa, k_init, first_conv_stride=(1,2,2))

    # DECODER
    for i in range(len(fa_save)):
        # tu
        if i == 0:
            x = Conv3DTranspose(fa_save[-(i+1)], (1, 2, 2), use_bias=False,
                                strides=(1, 2, 2), padding='valid') (x)
        else:
            x = Conv3DTranspose(fa_save[-(i+1)], (2, 2, 2), use_bias=False,
                                strides=(2, 2, 2), padding='valid') (x)
        x = concatenate([x, l[-(i+1)]])

        # conv_blocks_localization
        x = StackedConvLayers(x, fa_save[-(i+1)], k_init, first_conv_stride=1)
        seg_outputs.append(Conv3D(num_classes, (1, 1, 1), use_bias=False, activation=activation) (x))   

    outputs = seg_outputs
    
    model = Model(inputs=[inputs], outputs=[outputs])

    # Calculate the weigts as nnUNet does
    ################# Here we wrap the loss for deep supervision ############
    # we need to know the number of outputs of the network
    net_numpool = num_pool

    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss
    weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

    # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
    weights[~mask] = 0
    weights = weights / weights.sum()
    weights = weights[::-1] 
    ################# END ###################

    return model



def StackedConvLayers(x, feature_maps, k_init, first_conv_stride=2):
    x = ConvDropoutNormNonlin(x, feature_maps, k_init, first_conv_stride=first_conv_stride)
    x = ConvDropoutNormNonlin(x, feature_maps, k_init)
    return x

    
def ConvDropoutNormNonlin(x, feature_maps, k_init, first_conv_stride=1):
#     x = ZeroPadding3D(padding=(1, 1, 1))(x)
    x = Conv3D(feature_maps, (3, 3, 3), strides=first_conv_stride, activation=None,
               kernel_initializer=k_init, padding='valid') (x)
    x = BatchNormalization(epsilon=1e-05, momentum=0.1) (x)
    x = LeakyReLU(alpha=0.01) (x)
    return x