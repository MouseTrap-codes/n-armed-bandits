# exponential recency-weighted average (for nonstationary problems)
def exponential_recency_weighted_average_method(
    q_prev: float, reward: float, alpha: float
) -> float:
    return q_prev + (alpha) * (reward - q_prev)
