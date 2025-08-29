from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from n_armed_bandits.agents import EpsilonGreedyAgent
from n_armed_bandits.envs import NonstationaryTestbed

"""
Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary
problems. Use a modified version of the 10-armed testbed in which all the
q(a) start out equal and then take independent random walks. Prepare plots
like Figure 2.1 for an action-value method using sample averages, incrementally computed by α =
1/k, and another action-value method using a constant step-size parameter, α = 0.1. Use ε = 0.1 and, if necessary, runs longer than
1000 plays.

Caption from Sutton and Barto
"""

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# setup
num_runs = 2000
num_steps = 10000
n = 10

# accumulators
avg_rewards_sample_avg = np.zeros(num_steps)
avg_rewards_weighted_avg = np.zeros(num_steps)
optimal_action_sample_avg = np.zeros(num_steps)
optimal_action_weighted_avg = np.zeros(num_steps)

for run in range(num_runs):
    testbed = NonstationaryTestbed(n=n)
    agents = [
        EpsilonGreedyAgent(n=n, epsilon=0.1),
        EpsilonGreedyAgent(n=n, epsilon=0.1, alpha=0.1),
    ]

    for t in range(num_steps):
        for i, agent in enumerate(agents):
            action = agent.select_action()
            reward = testbed.get_reward(action)
            agent.update(action, reward)

            if i == 0:
                avg_rewards_sample_avg[t] += reward
                optimal_action_sample_avg[t] += action == testbed.get_optimal_action()
            elif i == 1:
                avg_rewards_weighted_avg[t] += reward
                optimal_action_weighted_avg[t] += action == testbed.get_optimal_action()


# take averages over the runs
avg_rewards_sample_avg /= num_runs
avg_rewards_weighted_avg /= num_runs
optimal_action_sample_avg = (optimal_action_sample_avg / num_runs) * 100
optimal_action_weighted_avg = (optimal_action_weighted_avg / num_runs) * 100

# plot!
plt.figure(figsize=(12, 5))

# average rewards over time
plt.subplot(1, 2, 1)
plt.plot(avg_rewards_sample_avg, label="ε-Greedy (α = 1/k, ε = 0.1)")
plt.plot(avg_rewards_weighted_avg, label="ε-Greedy (α = 0.1, ε = 0.1)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time")
plt.legend()

# % optimal action over time
# % Optimal Action subplot
plt.subplot(1, 2, 2)
plt.plot(optimal_action_sample_avg, label="ε-Greedy (α = 1/k, ε = 0.1)")
plt.plot(optimal_action_weighted_avg, label="ε-Greedy (α = 0.1, ε = 0.1)")
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("% Optimal Action over Time")
plt.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "exercise_two_four.png", dpi=300, bbox_inches="tight")
plt.show()
