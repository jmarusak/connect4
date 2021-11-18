#!/usr/bin/env python3

import math

from c4.game import Game
from c4.agents import MCTSNode
from c4.agents import show_tree
from c4.player import Player
from c4.agents import RandomAgent
from c4.utils import print_board

game = Game.new_game(board_size=(4,4))
root = MCTSNode(game)


def simulate_random_game(game):
        agents = {
            Player.x: RandomAgent(),
            Player.o: RandomAgent(),
        }
        while not game.is_over():
            move = agents[game.next_player].select_move(game)
            game = game.apply_move(move)
        return game.winner()

def select_child(node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None
        # Loop over each child.
        for child in node.children:
            # Calculate the UCT score.
            win_percentage = child.winning_frac(node.game.next_player)
            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + 2 * exploration_factor
            # Check if this is the largest we've seen so far.
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

for i in range(100):
    node = root

    while (not node.can_add_child()) and (not node.is_terminal()):
        node = select_child(node)

    if node.can_add_child():
        node = node.add_random_child()

    print_board(node.game.board)

    winner = simulate_random_game(node.game)
    if winner == None:
        winner = Player.o

    while node is not None:
        node.record_win(winner)
        node = node.parent 

show_tree(root)
