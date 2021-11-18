import copy

from c4.player import Player
from c4.board import Board
from c4.point import Point

class Game:
    def __init__(self, board, watermark, next_player):
        self.board = board
        self.watermark = watermark
        self.next_player = next_player

    def apply_move(self, move):
        col = move - 1
        row = self.watermark[col]
        next_board = copy.deepcopy(self.board)
        next_board.drop_stone(self.next_player, Point(row=row, col=col))
        
        next_watermark = copy.deepcopy(self.watermark)
        if next_watermark[col] < self.board.num_rows-1:
            next_watermark[col] += 1
        else:
            next_watermark[col] = None
        return Game(next_board, next_watermark, self.next_player.other)

    @classmethod
    def new_game(cls, board_size):
        board = Board(*board_size)
        watermark = [0 for i in range(board.num_cols)]
        return Game(board, watermark, Player.x)

    def legal_moves(self):
        moves = [col+1 for col in range(self.board.num_cols) if self.watermark[col] is not None]
        return moves

    def is_over(self):
        if self._has_4_in_row(Player.x):
            return True 
        if self._has_4_in_row(Player.o):
            return True
        if len(self.legal_moves()) == 0:
            return True
        return False

    def winner(self):
        if self._has_4_in_row(Player.x):
            return Player.x
        if self._has_4_in_row(Player.o):
            return Player.o
        return None

    def _has_4_in_row(self, player):
        for row in range(self.board.num_rows):
            for col in range(self.board.num_cols - 3):
                if self.board.get(Point(row, col)) == player and \
                   self.board.get(Point(row, col+1)) == player and \
                   self.board.get(Point(row, col+2)) == player and \
                   self.board.get(Point(row, col+3)) == player:
                    return True
        for row in range(self.board.num_rows-2):
            for col in range(self.board.num_cols):
                if self.board.get(Point(row, col)) == player and \
                   self.board.get(Point(row+1, col)) == player and \
                   self.board.get(Point(row+2, col)) == player and \
                   self.board.get(Point(row+3, col)) == player:
                    return True
        for row in range(self.board.num_rows-3):
            for col in range(self.board.num_cols-3):
                if self.board.get(Point(row, col)) == player and \
                   self.board.get(Point(row+1, col+1)) == player and \
                   self.board.get(Point(row+2, col+2)) == player and \
                   self.board.get(Point(row+3, col+3)) == player:
                    return True
        for row in range(3, self.board.num_rows):
            for col in range(self.board.num_cols-3):
                if self.board.get(Point(row, col)) == player and \
                   self.board.get(Point(row-1, col+1)) == player and \
                   self.board.get(Point(row-2, col+2)) == player and \
                   self.board.get(Point(row-3, col+3)) == player:
                    return True
        return False
