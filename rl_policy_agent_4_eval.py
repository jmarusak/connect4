#!/usr/bin/env python3

from collections import namedtuple

from keras.models import load_model

from c4.game import Game
from c4.player import Player
from c4.agents import PolicyAgent
from c4.encoders import OnePlaneEncoder
from c4.utils import print_board, print_encoded_board

from c4.rl import ExperienceCollector, combine_experience

NUM_GAMES=100

BOARD_NUM_ROWS=6
BOARD_NUM_COLS=7

class GameRecord(namedtuple('GameRecord', 'moves winner')):
    pass

def simulate_game(board_size, agent_x, agent_o):
    moves = []
    game = Game.new_game(board_size)

    while not game.is_over():

        agents = {
            Player.x: agent_x,
            Player.o: agent_o
        }    
        
        move = agents[game.next_player].select_move(game)
        if move == None:
            break
        moves.append(move)
        game = game.apply_move(move)

    winner = game.winner()
    
    print_board(game.board)
    print('Winner:', winner)

    return GameRecord(
        moves=moves,
        winner=winner
    )
        
def main():
    board_size = (BOARD_NUM_ROWS, BOARD_NUM_COLS)    

    encoder = OnePlaneEncoder(board_size)
    
    # latest model
    model_x = load_model('./c4/models/rl_policy_agent_model.h5')
    agent_x = PolicyAgent(model_x, encoder)
    
    # previous model
    model_o = load_model('./c4/models/rl_policy_agent_model_previous.h5')
    agent_o = PolicyAgent(model_o, encoder)

    wins = 0
    losses = 0
    for i in range(NUM_GAMES):
        game_record = simulate_game(board_size, agent_x, agent_o)
        print('Simulating game {}/{}'.format(i+1, NUM_GAMES))
        
        if game_record.winner == Player.x:
            wins += 1
        else:
            losses += 1
    print('\nAgent x wins: {}/{}'.format(wins, wins+losses))
    
if __name__ == '__main__':
    main()                
