import numpy as np

class KArmedBandit:
    def __init__(self, k):
        self.k = k
        self.q_true = np.random.normal(0, 1, k)  # True action values
        self.optimal_action = np.argmax(self.q_true)  # Optimal action
        
        self.action_count = np.zeros(k)  # Count of actions taken
        self.q_estimated = np.zeros(k)  # Estimated action values

    def step(self, action):
        reward = np.random.normal(self.q_true[action], 1)
        self.action_count[action] += 1
        
        # sample-average method to update estimated action value
        self.q_estimated[action] += (reward - self.q_estimated[action]) / self.action_count[action]
        
        return reward