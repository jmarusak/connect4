from keras.layers import Dense, Flatten
from keras.layers import Conv2D, ZeroPadding2D


def layers(input_shape):
    return [
        ZeroPadding2D(padding=(3,3), input_shape=input_shape),
        Conv2D(48, kernel_size=(4,4), activation='relu'),
        ZeroPadding2D(padding=(3,3)),
        Conv2D(32, kernel_size=(4,4), activation='relu'),
        ZeroPadding2D(padding=(3,3)),
        Conv2D(16, kernel_size=(4,4), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
    ]
