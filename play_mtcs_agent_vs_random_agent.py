#!/usr/bin/env python3

import time

from c4.player import Player
from c4.game import Game
from c4.utils import print_board

from c4.agents import RandomAgent, MCTSAgent

BOARD_NUM_ROWS=6
BOARD_NUM_COLS=7

def main():
    board_size = (BOARD_NUM_ROWS, BOARD_NUM_COLS)    

    agents = {
        Player.x: MCTSAgent(num_rounds=700, temperature=0.5),
        Player.o: RandomAgent()
    }

    game = Game.new_game(board_size=board_size)
    
    while not game.is_over():
        #time.sleep(0.2)
        print(chr(27) + "[2J")
        print_board(game.board)
         
        move = agents[game.next_player].select_move(game)
        game = game.apply_move(move)
    
    print(chr(27) + "[2J")
    print_board(game.board)
    print('Winner: ', game.winner())
    
if __name__ == '__main__':
    main()                
