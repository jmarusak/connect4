#!/usr/bin/env python3

import unittest

from c4.board import Board
from c4.game import Game
from c4.player import Player
from c4.point import Point
from c4.utils import print_board

class GameTest(unittest.TestCase):
    def test_4_in_row(self):
        moves = [Point(5,0), Point(4,1), Point(3,2), Point(2,3)]

        game = Game.new_game(board_size=(6,7))
        for move in moves:
            game.board.drop_stone(Player.x, move)
 
        print_board(game.board)
        self.assertTrue(game._has_4_in_row(player=Player.x))
       
if __name__ == '__main__':
    unittest.main() 
