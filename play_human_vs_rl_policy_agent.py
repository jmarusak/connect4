#!/usr/bin/env python3

from c4.player import Player
from c4.game import Game
from c4.utils import print_board

from c4.agents import PolicyAgent
from c4.encoders import OnePlaneEncoder

from keras.models import load_model

BOARD_NUM_ROWS=6
BOARD_NUM_COLS=7

def main():
    board_size = (BOARD_NUM_ROWS, BOARD_NUM_COLS)
  
    model = load_model('./c4/models/rl_policy_agent_model.h5')
    encoder = OnePlaneEncoder(board_size) 
     
    agent = PolicyAgent(model, encoder)
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
