import numpy as np

def normalize(policy):
    policy = np.clip(policy, 0, 1)
    return policy / np.sum(policy)

def simulate_game(actions, policy):
    p1_actions = {1:0, 2:0, 3:0, 4:0, 5:0}
    p2_actions = {1:0, 2:0, 3:0, 4:0, 5:0}
    
    p1_state = 0
    p2_state = 0

    for _ in range(100):
        p1_action = np.random.choice(actions, p=policy)
        p1_actions[p1_action] += 1
        p1_state += p1_action

        p2_action = np.random.choice(actions, p=policy)
        p2_actions[p2_action] += 1
        p2_state += p2_action
        
    if p1_state > p2_state:
        win_actions = p1_actions
        lose_actions = p2_actions
    else:
        win_actions = p2_actions
        lose_actions = p1_actions
        
    return (win_actions, lose_actions)

def main():
    num_games = 1000
    learning_rate = 0.0001
    actions = [1, 2, 3, 4, 5]
    policy = [0.2, 0.2, 0.2, 0.2, 0.2]

    for i in range(num_games):
        win_actions, lose_actions = simulate_game(actions, policy)
        
        for index, action in enumerate(actions):
            net_wins = win_actions[action] - lose_actions[action]
            policy[index] += learning_rate * net_wins
        
        if i % 100 == 0:
            policy = normalize(policy)
            print('{}: {}'.format(i, policy))

if __name__ == '__main__':
    main()
