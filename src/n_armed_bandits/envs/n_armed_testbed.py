import numpy as np


# n-armed testbed
class NArmedTestbed:
    def __init__(self, n: int, mean: float = 0.0, std: float = 1.0):
        # intialize true action values using normal distribution
        self.q_star: np.ndarray = np.random.normal(loc=mean, scale=std, size=n)
        self.n = n

    def get_reward(self, action: int):
        # return noisy reward
        return self.q_star[action] + np.random.normal(loc=0.0, scale=1.0)

    def get_optimal_action(self):
        return np.argmax(self.q_star)
