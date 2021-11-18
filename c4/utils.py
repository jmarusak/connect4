import numpy as np
from c4.point import Point
from c4.player import Player


def print_board(board):
    STONE_TO_CHAR = {
        None: ' . ',
        Player.x: ' x ',
        Player.o: ' o ',
    }

    print('\n ' + '  '.join([str(col+1) for col in range(board.num_cols)]))
    for row in range(board.num_rows-1, -1, -1):
        line = []
        for col in range(board.num_cols):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print(''.join(line))


def print_encoded_board(board):
    STONE_TO_CHAR = {
        0: ' . ',
        1: ' x ',
        -1: ' o ',
    }

    num_rows=6
    num_cols=7

    board = np.reshape(board, newshape=(num_rows, num_cols))

    print('\n ' + '  '.join([str(col+1) for col in range(num_cols)]))
    for row in range(num_rows-1, -1, -1):
        line = []
        for col in range(num_cols):
            stone = board[row, col]
            line.append(STONE_TO_CHAR[stone])
        print(''.join(line))
