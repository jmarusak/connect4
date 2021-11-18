import importlib

class Encoder:
    def name(self):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()
    
    def encode(self, game):
        raise NotImplementedError()

def get_encoder_by_name(name, board_size):
    module = importlib.import_module('c4.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(board_size)
