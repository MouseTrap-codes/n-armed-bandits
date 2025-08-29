import matplotlib.pyplot as plt
import numpy as np

from n_armed_bandits.agents import GradientBanditAgent
from n_armed_bandits.envs import NArmedTestbed

"""
Figure 2.4: Average performance of the gradient-bandit algorithm with and
without a reward baseline on the 10-armed testbed with E[q(a)] = 4.

Caption from Sutton and Barto
"""

# setup
num_runs = 2000
num_steps = 1000
n = 10

# accumulators
optimal_action_01_with_baseline = np.zeros(num_steps)
optimal_action_01_without_baseline = np.zeros(num_steps)
optimal_action_04_with_baseline = np.zeros(num_steps)
optimal_action_04_without_baseline = np.zeros(num_steps)


for run in range(num_runs):
    testbed = NArmedTestbed(n=n, mean=4.0)
    agents = [
        # a = 0.1
        GradientBanditAgent(n=n, alpha=0.1, use_baseline=True),
        GradientBanditAgent(n=n, alpha=0.1, use_baseline=False),
        # a = 0.4
        GradientBanditAgent(n=n, alpha=0.4, use_baseline=True),
        GradientBanditAgent(n=n, alpha=0.4, use_baseline=False),
    ]

    for t in range(num_steps):
        for i, agent in enumerate(agents):
            action = agent.select_action()
            reward = testbed.get_reward(action)
            agent.update(action, reward)

            if i == 0:
                optimal_action_01_with_baseline[t] += (
                    action == testbed.get_optimal_action()
                )
            elif i == 1:
                optimal_action_01_without_baseline[t] += (
                    action == testbed.get_optimal_action()
                )
            elif i == 2:
                optimal_action_04_with_baseline[t] += (
                    action == testbed.get_optimal_action()
                )
            elif i == 3:
                optimal_action_04_without_baseline[t] += (
                    action == testbed.get_optimal_action()
                )


# calculate % optimal action
optimal_action_01_with_baseline = (optimal_action_01_with_baseline / num_runs) * 100
optimal_action_01_without_baseline = (
    optimal_action_01_without_baseline / num_runs
) * 100
optimal_action_04_with_baseline = (optimal_action_04_with_baseline / num_runs) * 100
optimal_action_04_without_baseline = (
    optimal_action_04_without_baseline / num_runs
) * 100


# plot!
# average rewards over time
plt.figure(figsize=(12, 5))
plt.plot(
    optimal_action_01_with_baseline, label="Gradient Bandit (a = 0.1, with baseline)"
)
plt.plot(
    optimal_action_01_without_baseline,
    label="Gradient Bandit (a = 0.1, without baseline)",
)
plt.plot(
    optimal_action_04_with_baseline, label="Gradient Bandit (a = 0.4, with baseline)"
)
plt.plot(
    optimal_action_04_without_baseline,
    label="Gradient Bandit (a = 0.4, without baseline)",
)
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("% Optimal Action over Time")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
