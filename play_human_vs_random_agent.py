#!/usr/bin/env python3

from c4.player import Player
from c4.game import Game
from c4.utils import print_board

from c4.agents import RandomAgent

BOARD_NUM_ROWS=4
BOARD_NUM_COLS=4

def main():
    board_size = (BOARD_NUM_ROWS, BOARD_NUM_COLS)
    
    agent = RandomAgent()
    game = Game.new_game(board_size=board_size)
    
    while not game.is_over():
        print(chr(27) + "[2J")
        print_board(game.board)
        
        if(game.next_player == Player.x):
            human_move = input('-- ')
            move = int(human_move) 
        else:
            move = agent.select_move(game)
        game = game.apply_move(move)
    
    print(chr(27) + "[2J")
    print_board(game.board)
    print('Winner: ', game.winner())
    
if __name__ == '__main__':
    main()                
