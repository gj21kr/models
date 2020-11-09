from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.models import Model

def cdc(input_shape, classes, levels=3):
    inputs = Input(shape=input_shape)
    x = inputs
    for level in range(levels):
        x = Conv2D(filters=3, kernel_size=(3,3), padding='valid')(x)
        x = Conv2D(filters=3, kernel_size=(3,3), padding='valid')(x)
        x = MaxPooling2D(pool_size=(2,2), padding='valid')(x)

#     x = Conv2D(filters=3, kernel_size=(3,3), padding='valid')(x)
#     x = Conv2D(filters=3, kernel_size=(3,3), padding='valid')(x)
#     x = Conv2D(filters=3, kernel_size=(3,3), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(classes)(x)

    model = Model(inputs=inputs, outputs=x)

    return model