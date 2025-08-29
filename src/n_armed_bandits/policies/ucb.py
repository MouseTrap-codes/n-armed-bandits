import numpy as np


# upper-confidence-bound action selection -> select among non-greedy actions according to their potential for
# actually being optimal: how close estimated are to being maximal and uncertainties in those estimates
def ucb_select(q_t_values: np.ndarray, n_t_values: np.ndarray, c: float) -> int:
    # check for unvisited actions first
    unvisited = np.where(n_t_values == 0)[0]
    if unvisited.size > 0:
        return unvisited[0]

    # standard ucb calculation
    return np.argmax(
        q_t_values + c * np.sqrt(np.log(np.sum(n_t_values)) / n_t_values)
    ).item()
