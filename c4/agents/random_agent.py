import random

from c4.agents import Agent

class RandomAgent(Agent):
    def select_move(self, game):
        legal_moves = game.legal_moves()
        return random.choice(legal_moves)        
