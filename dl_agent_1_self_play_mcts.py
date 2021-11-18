#!/usr/bin/env python3

import numpy as np

from c4.game import Game
from c4.agents import MCTSAgent
from c4.encoders import get_encoder_by_name
from c4.utils import print_board

NUM_GAMES=50

BOARD_NUM_ROWS=6
BOARD_NUM_COLS=7

MCTS_ROUNDS=100
MCTS_TEMPERATURE=1.4

FILE_NAME_FEATURES='./data/features.npy'
FILE_NAME_LABELS='./data/labels.npy'

def generate_game_data(board_size, rounds, temperature):
    boards = []
    moves = []

    game = Game.new_game(board_size)
    agent = MCTSAgent(rounds, temperature)
    encoder = get_encoder_by_name('oneplane_encoder', board_size)

    while not game.is_over():
        # Collect boards 
        boards.append(encoder.encode(game))
    
        # Collect moves 
        move = agent.select_move(game)
        move_one_hot_encoded = np.zeros(BOARD_NUM_COLS)
        move_one_hot_encoded[move-1] = 1
        moves.append(move_one_hot_encoded)

        game = game.apply_move(move)

        print(chr(27) + "[2J")
        print_board(game.board)

    return np.array(boards), np.array(moves)
        
def main():
    board_size = (BOARD_NUM_ROWS, BOARD_NUM_COLS)    

    Xs = []  # features
    ys = []  # labels
    for i in range(NUM_GAMES):
        X, y = generate_game_data(board_size, MCTS_ROUNDS, MCTS_TEMPERATURE)
        print('Generated game {}/{}'.format(i+1, NUM_GAMES))
        Xs.append(X)
        ys.append(y)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)

    print(X.shape)
    print(y.shape)

    # Save to files
    np.save(FILE_NAME_FEATURES, X)
    np.save(FILE_NAME_LABELS, y)
    
        
if __name__ == '__main__':
    main()                
