#!/usr/bin/env python3

import numpy as np
from keras.models import load_model, save_model
from tensorflow.keras.optimizers import SGD

from c4.rl import load_experience

BOARD_NUM_ROWS=5
BOARD_NUM_COLS=5

def main():

    model = load_model('./c4/models/rl_policy_agent_model.h5')

    optimizer = SGD(learning_rate=0.00001, clipnorm=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    experience = load_experience('./c4/models/rl_policy_agent_experience.h5')

    n = experience.states.shape[0]
    print(n)
    num_moves = BOARD_NUM_COLS

    X = experience.states
    y = np.zeros((n, num_moves))
    for i in range(n):
        action = experience.actions[i]
        reward = experience.rewards[i]
        y[i][action-1] = reward

    model.fit(
       X, y,
       batch_size=128,
       epochs=1)

    save_model(model, './c4/models/rl_policy_agent_model.h5')

if __name__ == '__main__':
    main()
