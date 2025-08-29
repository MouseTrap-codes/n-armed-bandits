import numpy as np


# nonstationary testbed
class NonstationaryTestbed:
    def __init__(self, n: int, mean: float = 0.0, std: float = 1.0):
        # intialize true action values using normal distribution
        self.q_star = np.random.normal(loc=mean, scale=std, size=n)
        self.n = n

    def get_reward(self, action: int):
        reward = self.q_star[action] + np.random.normal(loc=0.0, scale=1.0)

        # change all q values randomly
        self.q_star += np.random.normal(loc=0.0, scale=0.01, size=self.n)

        # return noisy reward
        return reward

    def get_optimal_action(self):
        return np.argmax(self.q_star)
