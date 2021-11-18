from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout


def layers(input_shape):
    return [
        Conv2D(42, kernel_size=(3,3), activation='relu', padding='same', input_shape=input_shape),
        #Dropout(rate=0.5),
        Conv2D(42, kernel_size=(3,3), activation='relu', padding='same'),
        #MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(512, activation='relu'),
    ]
