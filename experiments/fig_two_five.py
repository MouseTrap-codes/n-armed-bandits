from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from n_armed_bandits.agents import EpsilonGreedyAgent, GradientBanditAgent, UCBAgent
from n_armed_bandits.agents.base import Agent
from n_armed_bandits.envs import NArmedTestbed

"""
Figure 2.5: A parameter study of the various bandit algorithms presented in
this chapter. Each point is the average reward obtained over 1000 steps with
a particular algorithm at a particular setting of its parameter.

Caption from Sutton and Barto
"""

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# parameters
epsilons = 1 / (2.0 ** np.arange(7, 0, -1))  # [1/128 to 1]
cs = 1 / (2.0 ** np.arange(4, -3, -1))  # [1/16 to 4]
alphas = 1 / (2.0 ** np.arange(5, -3, -1))  # [1/32 to 4]
q0s = 1 / (2.0 ** np.arange(3, -3, -1))  # [1/8 to 4]

# values to store
average_reward_e_greedy = []
average_reward_ucb = []
average_reward_gradient = []
average_reward_optimistic_greedy = []

# setup
num_runs = 2000
num_steps = 1000
n = 10

# e-greedy
for eps in epsilons:
    rewards = []
    for run in range(num_runs):
        testbed = NArmedTestbed(n=n)
        agent: Agent = EpsilonGreedyAgent(n=n, epsilon=eps)
        run_reward = 0.0
        for step in range(num_steps):
            action = agent.select_action()
            reward = testbed.get_reward(action)
            agent.update(action, reward)
            run_reward += reward
        rewards.append(run_reward / num_steps)
    average_reward_e_greedy.append(np.mean(rewards))

# ucb
for c in cs:
    rewards = []
    for run in range(num_runs):
        testbed = NArmedTestbed(n=n)
        agent = UCBAgent(n=n, c=c)
        run_reward = 0.0
        for step in range(num_steps):
            action = agent.select_action()
            reward = testbed.get_reward(action)
            agent.update(action, reward)
            run_reward += reward
        rewards.append(run_reward / num_steps)
    average_reward_ucb.append(np.mean(rewards))

# gradient
for alpha in alphas:
    rewards = []
    for run in range(num_runs):
        testbed = NArmedTestbed(n=n)
        agent = GradientBanditAgent(n=n, alpha=alpha)
        run_reward = 0.0
        for step in range(num_steps):
            action = agent.select_action()
            reward = testbed.get_reward(action)
            agent.update(action, reward)
            run_reward += reward
        rewards.append(run_reward / num_steps)
    average_reward_gradient.append(np.mean(rewards))

# optimistic greedy
for q0 in q0s:
    rewards = []
    for run in range(num_runs):
        testbed = NArmedTestbed(n=n)
        agent = EpsilonGreedyAgent(n=n, epsilon=0.0, alpha=0.1, initial_estimates=q0)
        run_reward = 0.0
        for step in range(num_steps):
            action = agent.select_action()
            reward = testbed.get_reward(action)
            agent.update(action, reward)
            run_reward += reward
        rewards.append(run_reward / num_steps)
    average_reward_optimistic_greedy.append(np.mean(rewards))

# plot!
plt.figure(figsize=(10, 6))
plt.plot(epsilons, average_reward_e_greedy, label=r"$\epsilon$-greedy")
plt.plot(cs, average_reward_ucb, label="UCB")
plt.plot(alphas, average_reward_gradient, label="Gradient Bandit")
plt.plot(
    q0s,
    average_reward_optimistic_greedy,
    label="greedy with\noptimistic\ninitialization\n$\\alpha = 0.1$",
)

x_ticks = [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]
x_labels = ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]
plt.xscale("log", base=2)
plt.xticks(x_ticks, x_labels)
plt.xlabel(r"$\epsilon \; / \alpha \; / \; c \; / \; Q_0$", fontsize=14)
plt.ylabel("Average reward over first 1000 steps", fontsize=12)


plt.ylim(1.0, 1.5)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.legend()

plt.tight_layout()
plt.title("Figure 2.5: Parameter Study of Bandit Algorithms")
plt.savefig(RESULTS_DIR / "fig_two_five.png", dpi=300, bbox_inches="tight")
plt.show()
