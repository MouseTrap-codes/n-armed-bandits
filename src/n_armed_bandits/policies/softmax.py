import numpy as np


def softmax(h: np.ndarray) -> np.ndarray:
    exp_values = np.exp(h - np.max(h))
    return exp_values / np.sum(exp_values)


def softmax_sample(pi: np.ndarray) -> int:
    # choose action by softmax probability
    return np.random.choice(len(pi), p=pi)
