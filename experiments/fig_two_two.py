import matplotlib.pyplot as plt
import numpy as np

from n_armed_bandits.agents import EpsilonGreedyAgent
from n_armed_bandits.envs import NArmedTestbed

"""
Figure 2.2: The effect of optimistic initial action-value estimates on the 10-
armed testbed. Both methods used a constant step-size parameter, α = 0.1.

Caption from Sutton and Barto
"""

# setup
num_runs = 2000
num_steps = 1000
n = 10

# accumulators
optimal_action_optimistic = np.zeros(num_steps)
optimal_action_realistic = np.zeros(num_steps)

for run in range(num_runs):
    testbed = NArmedTestbed(n=n)
    agents = [
        # optimistic, greedy
        EpsilonGreedyAgent(n=n, alpha=0.1, epsilon=0.0, initial_estimates=5.0),
        # realistic, ε-greedy
        EpsilonGreedyAgent(n=n, alpha=0.1, epsilon=0.1),
    ]
    rewards = [0.0, 0.0, 0.0]

    for t in range(num_steps):
        for i, agent in enumerate(agents):
            action = agent.select_action()
            reward = testbed.get_reward(action)
            agent.update(action, reward)

            if i == 0:
                optimal_action_optimistic[t] += action == testbed.get_optimal_action()
            elif i == 1:
                optimal_action_realistic[t] += action == testbed.get_optimal_action()

# calculate % optimal action
optimal_action_optimistic = (optimal_action_optimistic / num_runs) * 100
optimal_action_realistic = (optimal_action_realistic / num_runs) * 100

# plot!
plt.figure(figsize=(6, 4))
plt.plot(optimal_action_optimistic, label="Optimistic Greedy (Q₁=5, ε = 0)")
plt.plot(optimal_action_realistic, label="Realistic ε-Greedy (Q₁=0, ε = 0.1)")
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("% Optimal Action over Time")
plt.legend()

plt.tight_layout()
plt.show()
