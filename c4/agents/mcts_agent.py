import math
import random

from c4.agents import Agent, RandomAgent
from c4.player import Player

class MCTSNode(object):
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.x: 0,
            Player.o: 0
        }
        self.children = []
        self.num_rollouts = 0
        self.unvisited_moves = game.legal_moves()

    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game = self.game.apply_move(new_move)
        new_node = MCTSNode(new_game, self, new_move)
        self.children.append(new_node)
        return new_node

    def is_terminal(self):
        return self.game.is_over()

    def record_win(self, player):
        self.win_counts[player] += 1
        self.num_rollouts += 1

    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts) 

class MCTSAgent(Agent):
    def __init__(self, num_rounds, temperature):
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game):
        root = MCTSNode(game)

        for i in range(self.num_rounds):
            node = root
        
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            if node.can_add_child():
                node = node.add_random_child()

            winner = self.simulate_random_game(node.game)
            if winner == None:
                winner = Player.o

            while node is not None:
                node.record_win(winner)
                node = node.parent

        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(game.next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        #print('Select move %s with win pct %.3f' % (best_move, best_pct))
        return best_move     


    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None
        # Loop over each child.
        for child in node.children:
            # Calculate the UCT score.
            win_percentage = child.winning_frac(node.game.next_player)
            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + self.temperature * exploration_factor
            # Check if this is the largest we've seen so far.
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child


    def simulate_random_game(self, game):
        agents = {
            Player.x: RandomAgent(),
            Player.o: RandomAgent(),
        }   
        while not game.is_over():
            move = agents[game.next_player].select_move(game)
            game = game.apply_move(move)
        return game.winner()


def show_tree(node, indent='', max_depth=3):
    if max_depth < 0:
        return
    if node is None:
        return
    if node.parent is None:
        print('%sroot' % indent)
    else:
        player = node.parent.game.next_player
        move = node.move
        print('%s%s %s %d %d' % (
            indent, player, move,
            node.num_rollouts,
            node.win_counts[player],
        ))
    for child in sorted(node.children, key=lambda n: n.num_rollouts, reverse=True):
        show_tree(child, indent + '  ', max_depth - 1)
