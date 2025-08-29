import matplotlib.pyplot as plt
import numpy as np

from n_armed_bandits.agents import EpsilonGreedyAgent, UCBAgent
from n_armed_bandits.envs import NArmedTestbed

"""
Figure 2.3: Average performance of UCB action selection on the 10-armed
testbed. As shown, UCB generally performs better that ε-greedy action selection, except in the first n plays, when it selects randomly among the as-yetunplayed actions. UCB with c = 1 would perform even better but would not
show the prominent spike in performance on the 11th play. Can you think of
an explanation of this spike?

Caption from Sutton and Barto
"""

# setup
num_runs = 2000
num_steps = 1000
n = 10

# accumulators
average_rewards_ucb = np.zeros(num_steps)
average_rewards_e_greedy = np.zeros(num_steps)

for run in range(num_runs):
    testbed = NArmedTestbed(n=n)
    agents = [
        # UCB Agent
        UCBAgent(n=n, c=2),
        # ε-greedy
        EpsilonGreedyAgent(n=n, epsilon=0.1),
    ]

    for t in range(num_steps):
        for i, agent in enumerate(agents):
            action = agent.select_action()
            reward = testbed.get_reward(action)
            agent.update(action, reward)

            if i == 0:
                average_rewards_ucb[t] += reward
            elif i == 1:
                average_rewards_e_greedy[t] += reward

# calculate % optimal action
average_rewards_ucb /= num_runs
average_rewards_e_greedy /= num_runs

# plot!
# average rewards over time
plt.figure(figsize=(6, 4))
plt.plot(average_rewards_ucb, label="UCB (c = 2)")
plt.plot(average_rewards_e_greedy, label="ε-Greedy (ε = 0.1)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time")
plt.legend()
plt.show()
