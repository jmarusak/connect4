import numpy as np
from c4.agents import Agent

class PolicyAgent(Agent):
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0

    def set_collector(self, collector):
        self.collector = collector

    def set_temperature(self, temperature):
        self.temperature = temperature

    def select_move(self, game):
        num_moves = self.encoder.num_cols
        move_candidates = np.arange(1, num_moves+1)
        
        board_matrix = self.encoder.encode(game)
      
        # transform to network input shape (NHWC)
        input_tensor = np.expand_dims(board_matrix, axis=0)
       
        if np.random.random() < self.temperature:
            # explore random moves
            move_probs = np.random.choice(move_candidates, num_moves)
        else:
            # follow our current policy.
            move_probs =  self.model.predict(input_tensor)[0]
        
        # clip probs
        eps = 1e-5
        min_prob = eps 
        max_prob = 1 - eps
        move_probs = np.clip(move_probs, min_prob, max_prob)
        # renormilize
        move_probs = move_probs / np.sum(move_probs)
        
        move_candidates = np.arange(1, num_moves+1)
        ranked_moves = np.random.choice(move_candidates, num_moves, p=move_probs)
        for move in ranked_moves:
            if move in game.legal_moves():
                if self.collector is not None:
                    self.collector.record_decision(state=board_matrix, action=move)
                return move
        return None
