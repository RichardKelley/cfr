import numpy as np

# To suppress scientific notation for our probabilities
np.set_printoptions(suppress=True)

# enums
ROCK = 0
PAPER = 1
SCISSORS = 2

NUM_ACTIONS = 3

class RPSTrainer:

    '''
    Use regret matching to learn how to play rock, paper, scissors
    against a fixed opponent.
    '''
    
    def __init__(self):
        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.strategy = np.zeros(NUM_ACTIONS)
        self.strategy_sum = np.zeros(NUM_ACTIONS)

        self.opp_strategy = np.asarray([0.4, 0.3, 0.3])

    def get_strategy(self):
        nonnegative = np.where(self.regret_sum > 0.0)
        self.strategy[nonnegative] = self.regret_sum[nonnegative] 
        normalizing_sum = np.sum(self.strategy)

        if normalizing_sum > 0.0:
            self.strategy /= normalizing_sum
        else:
            self.strategy = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
        self.strategy_sum += self.strategy
            
        return self.strategy

    def get_action(self, strat):
        return np.random.choice(np.arange(NUM_ACTIONS), 1, p = strat)

    def train(self, iterations):
        action_utility = np.zeros(NUM_ACTIONS)
        for i in range(iterations):
            strategy = self.get_strategy()
            my_action = self.get_action(strategy)
            other_action = self.get_action(self.opp_strategy)

            action_utility[other_action] = 0.0
            action_utility[0 if other_action == NUM_ACTIONS - 1 else other_action + 1] = 1
            action_utility[NUM_ACTIONS - 1 if other_action == 0 else other_action - 1] = -1
            
            self.regret_sum = action_utility - action_utility[my_action]
            
    def get_average_strategy(self):
        avg_strategy = np.zeros(NUM_ACTIONS)
        normalizing_sum = np.sum(self.strategy_sum)

        if normalizing_sum > 0.0:
            avg_strategy = self.strategy_sum / normalizing_sum
        else:
            avg_strategy = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
            
        return avg_strategy


if __name__ == '__main__':
    trainer = RPSTrainer()
    trainer.train(1000000)
    print(trainer.get_average_strategy())

