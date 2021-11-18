import numpy as np

from c4.point import Point
from c4.encoders import Encoder

class OnePlaneEncoder(Encoder):
    def __init__(self, board_size):
        self.num_rows, self.num_cols = board_size
        self.num_planes = 1

    def name(self):
        return 'oneplane_encoder'

    def shape(self):
        return self.num_rows, self.num_cols, self.num_planes

    def encode(self, game):
        # Encode next player stones as 1, opposite player -1
        board_matrix = np.zeros(self.shape())
        next_player = game.next_player
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                point = Point(row=row, col=col)
                player_on_board = game.board.get(point)
                
                if player_on_board is None:
                    continue
                if player_on_board == next_player:
                    board_matrix[row, col, 0] = 1
                else:
                    board_matrix[row, col, 0] = -1
        return board_matrix

def create(board_size):
    # Used to create the encoder by name
    return OnePlaneEncoder(board_size)
