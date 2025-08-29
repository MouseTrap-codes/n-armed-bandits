from typing import Optional

import numpy as np

from n_armed_bandits.policies import e_greedy
from n_armed_bandits.updates import (
    exponential_recency_weighted_average_method,
    sample_average_method,
)

from .base import Agent


class EpsilonGreedyAgent(Agent):
    def __init__(
        self,
        n: int,
        epsilon: float,
        alpha: Optional[float] = None,
        initial_estimates: float = 0.0,
    ):
        super().__init__(n)
        self.epsilon = epsilon
        self.q = np.full(n, initial_estimates)
        self.n_t = np.zeros(n)
        self.alpha = alpha  # None -> use sample average

    def select_action(self) -> int:
        return e_greedy(self.q, self.epsilon)

    def update(self, action: int, reward: float):
        self.n_t[action] += 1

        if self.alpha is None:
            self.q[action] = sample_average_method(
                q_prev=self.q[action], reward=reward, k=self.n_t[action]
            )
        else:
            self.q[action] = exponential_recency_weighted_average_method(
                q_prev=self.q[action], reward=reward, alpha=self.alpha
            )
