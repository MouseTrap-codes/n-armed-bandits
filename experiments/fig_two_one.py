from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from n_armed_bandits.agents import EpsilonGreedyAgent
from n_armed_bandits.envs import NArmedTestbed

"""
Figure 2.1: Average performance of ε-greedy action-value methods on the
10-armed testbed. These data are averages over 2000 tasks. All methods
used sample averages as their action-value estimates. The detailed structure
at the beginning of these curves depends on how actions are selected when
multiple actions have the same maximal action value. Here such ties were
broken randomly. An alternative that has a similar effect is to add a very
small amount of randomness to each of the initial action values, so that ties
effectively never happen.

Caption from Sutton and Bartos
"""
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# setup
num_runs = 2000
num_steps = 1000
n = 10

# accumulators
avg_rewards_01 = np.zeros(num_steps)
avg_rewards_001 = np.zeros(num_steps)
avg_rewards_0 = np.zeros(num_steps)
optimal_action_01 = np.zeros(num_steps)
optimal_action_001 = np.zeros(num_steps)
optimal_action_0 = np.zeros(num_steps)

for run in range(num_runs):
    testbed = NArmedTestbed(n=n)
    agents = [
        EpsilonGreedyAgent(n=n, epsilon=0.1),
        EpsilonGreedyAgent(n=n, epsilon=0.01),
        EpsilonGreedyAgent(n=n, epsilon=0.0),
    ]

    for t in range(num_steps):
        for i, agent in enumerate(agents):
            action = agent.select_action()
            reward = testbed.get_reward(action)
            agent.update(action, reward)

            if i == 0:
                avg_rewards_01[t] += reward
                optimal_action_01[t] += action == testbed.get_optimal_action()
            elif i == 1:
                avg_rewards_001[t] += reward
                optimal_action_001[t] += action == testbed.get_optimal_action()
            elif i == 2:
                avg_rewards_0[t] += reward
                optimal_action_0[t] += action == testbed.get_optimal_action()

# take averages over the runs
avg_rewards_01 /= num_runs
avg_rewards_001 /= num_runs
avg_rewards_0 /= num_runs
optimal_action_01 = (optimal_action_01 / num_runs) * 100
optimal_action_001 = (optimal_action_001 / num_runs) * 100
optimal_action_0 = (optimal_action_0 / num_runs) * 100

# plot!
plt.figure(figsize=(12, 5))

# average rewards over time
plt.subplot(1, 2, 1)
plt.plot(avg_rewards_01, label="ε = 0.1")
plt.plot(avg_rewards_001, label="ε = 0.01")
plt.plot(avg_rewards_0, label="ε = 0 (greedy)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time")
plt.legend()

# % optimal action over time
# % Optimal Action subplot
plt.subplot(1, 2, 2)
plt.plot(optimal_action_01, label="ε = 0.1")
plt.plot(optimal_action_001, label="ε = 0.01")
plt.plot(optimal_action_0, label="ε = 0 (greedy)")
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("% Optimal Action over Time")
plt.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_two_one.png", dpi=300, bbox_inches="tight")
plt.show()
