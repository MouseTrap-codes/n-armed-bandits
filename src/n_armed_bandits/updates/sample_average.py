# sample average method
def sample_average_method(q_prev: float, reward: float, k: int) -> float:
    return q_prev + (1 / k) * (reward - q_prev)
