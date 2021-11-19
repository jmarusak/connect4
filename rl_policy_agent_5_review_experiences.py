#!/usr/bin/env python3

import numpy as np
from c4.rl import load_experience

BOARD_NUM_ROWS=5
BOARD_NUM_COLS=5

def main():
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

    print(X)
    print(y)
if __name__ == '__main__':
    main()
