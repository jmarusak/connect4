#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import save_model

from c4.networks.cnn_small import layers

BOARD_NUM_ROWS=6
BOARD_NUM_COLS=7

def main():
    np.random.seed(42)

    X = np.load('./data/features.npy')
    y = np.load('./data/labels.npy')

    num_samples = X.shape[0]
    num_train_samples = int(0.7 * num_samples)

    X_train, X_test = X[:num_train_samples], X[num_train_samples:]
    y_train, y_test = y[:num_train_samples], y[num_train_samples:]

    model = Sequential()
    for layer in layers((BOARD_NUM_ROWS, BOARD_NUM_COLS, 1)):
        model.add(layer)
    model.add(Dense(BOARD_NUM_COLS, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

    save_model(model, './c4/models/dl_agent_model.h5')

if __name__ == '__main__':
    main()