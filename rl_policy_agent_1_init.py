#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense
from keras.models import save_model

from c4.networks.cnn_medium import layers

BOARD_NUM_ROWS=5
BOARD_NUM_COLS=5

def main():
    model = Sequential()
    for layer in layers((BOARD_NUM_ROWS, BOARD_NUM_COLS, 1)):
        model.add(layer)
    model.add(Dense(BOARD_NUM_COLS, activation='softmax'))
    print(model.summary())

    save_model(model, './c4/models/rl_policy_agent_model.h5')

if __name__ == '__main__':
    main()
