#!/usr/bin/env python3

from collections import namedtuple

from keras.models import load_model

from c4.game import Game
from c4.player import Player
from c4.agents import PolicyAgent
from c4.encoders import OnePlaneEncoder
from c4.utils import print_board, print_encoded_board

from c4.rl import ExperienceCollector, combine_experience

NUM_GAMES=2000

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

    model = load_model('./c4/models/rl_policy_agent_model.h5')
    encoder = OnePlaneEncoder(board_size)

    agent_x = PolicyAgent(model, encoder)
    agent_o = PolicyAgent(model, encoder)

    collector_x = ExperienceCollector()
    collector_o = ExperienceCollector()

    agent_x.set_collector(collector_x)
    agent_o.set_collector(collector_o)

    for i in range(NUM_GAMES):
        collector_x.begin_episode()
        collector_o.begin_episode()

        game_record = simulate_game(board_size, agent_x, agent_o)
        print('Simulating game {}/{}'.format(i+1, NUM_GAMES))
        
        if game_record.winner == Player.x:
            collector_x.complete_episode(reward=1)
            collector_o.complete_episode(reward=-1)
        else:
            collector_x.complete_episode(reward=-1)
            collector_o.complete_episode(reward=1)

        if i % 1000 == 0:
            experience = combine_experience([collector_x, collector_o])
            experience.serialize('./c4/models/rl_policy_agent_experience.h5')
        
    experience = combine_experience([collector_x, collector_o])
    experience.serialize('./c4/models/rl_policy_agent_experience.h5')
        
if __name__ == '__main__':
    main()                
