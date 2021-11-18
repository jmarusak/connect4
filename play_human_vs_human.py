#!/usr/bin/env python3

from c4.player import Player
from c4.game import Game
from c4.utils import print_board

BOARD_NUM_ROWS=4
BOARD_NUM_COLS=4

def main():
    board_size = (BOARD_NUM_ROWS, BOARD_NUM_COLS)
    
    game = Game.new_game(board_size=board_size)
    
    while not game.is_over():
        print(chr(27) + "[2J")
        print_board(game.board)
        
        if(game.next_player == Player.x):
            human_move = input('x-- ')
        else:
            human_move = input('o-- ')
        
        move = int(human_move)
        game = game.apply_move(move)

    print(chr(27) + "[2J")
    print_board(game.board)
    print('Winner: ', game.winner())

if __name__ == '__main__':
    main()                
