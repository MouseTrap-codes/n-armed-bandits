import numpy as np

from n_armed_bandits.agents import EpsilonGreedyAgent


def test_action_in_range():
    agent = EpsilonGreedyAgent(n=10, epsilon=0.1)
    for _ in range(100):
        action = agent.select_action()
        assert 0 <= action < 10


def test_greedy_behavior_when_epsilon_zero():
    agent = EpsilonGreedyAgent(n=5, epsilon=0.0)
    agent.q = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
    action = agent.select_action()
    assert action == 2


def test_update_sample_average():
    agent = EpsilonGreedyAgent(n=1, epsilon=0.0)
    agent.q[0] = 0.0
    agent.n_t[0] = 0
    agent.update(0, reward=10)
    assert agent.q[0] == 10

    agent.update(0, reward=0)
    assert np.isclose(agent.q[0], 5.0)
