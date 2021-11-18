import numpy as np
from c4.agents import Agent

class DeepLearningAgent(Agent):
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder

    def predict(self, game):
        board_matrix = self.encoder.encode(game)
        # transform to network input shape (NHWC)
        input_tensor = np.expand_dims(board_matrix, axis=0)
        return self.model.predict(input_tensor)[0]

    def select_move(self, game):
        move_probs = self.predict(game)
        # return move with max prob
        return np.argmax(move_probs)+1
