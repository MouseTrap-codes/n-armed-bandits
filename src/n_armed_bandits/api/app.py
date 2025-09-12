import numpy as np
from flask import Flask, jsonify, render_template, request

from n_armed_bandits.agents.epsilon_greedy import EpsilonGreedyAgent
from n_armed_bandits.agents.gradient import GradientBanditAgent
from n_armed_bandits.agents.ucb import UCBAgent
from n_armed_bandits.envs import NArmedTestbed, NonstationaryTestbed

model_map = {"epsilon-greedy": "epsilon_greedy", "ucb": "ucb", "gradient": "gradient"}

env_map = {"stationary": "stationary", "nonstationary": "nonstationary"}


def to_int(x, d=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return d


def to_float(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def to_bool(x, d=False):
    if x is None:
        return d
    s = str(x).strip().lower()
    return s in ("1", "true", "on", "yes")


app = Flask(__name__)


def load_agent(model_key: str, n: int, params: dict):
    model = model_map.get((model_key or "").strip().lower())
    if model == "epsilon_greedy":
        return EpsilonGreedyAgent(
            n=n,
            epsilon=to_float(params.get("epsilon"), 0.1),
            alpha=to_float(params.get("alpha")),  # None => sample-average updates
            initial_estimates=to_float(params.get("initial_estimates"), 0.0),
        )
    elif model == "ucb":
        return UCBAgent(
            n=n,
            c=to_float(params.get("ucb_c"), 2.0),
            alpha=to_float(params.get("alpha")),  # optional constant-α
        )
    elif model == "gradient":
        return GradientBanditAgent(
            n=n,
            alpha=to_float(params.get("alpha"), 0.1),
            use_baseline=to_bool(params.get("use_baseline"), True),
        )
    raise ValueError(f"Unknown model: {model}")


def load_env(env_key: str, n: int, mean: float, std: float):
    env = env_map.get((env_key or "").strip().lower())
    if env == "stationary":
        return NArmedTestbed(n=n, mean=mean, std=std)
    elif env == "nonstationary":
        return NonstationaryTestbed(n=n, mean=mean, std=std)
    raise ValueError(f"Unknown environment: {env}")


def run_simulation(
    model_key: str,
    env_key: str,
    num_arms: int,
    num_steps: int,
    mean_q: float,
    std_q: float,
    params: dict,
    num_runs: int = 200,
):
    avg_rewards = np.zeros(num_steps)
    pct_optimal_action = np.zeros(num_steps)

    for run in range(num_runs):
        agent = load_agent(model_key, num_arms, params)
        testbed = load_env(env_key, num_arms, mean_q, std_q)
        for t in range(num_steps):
            action = agent.select_action()
            reward = testbed.get_reward(action)
            agent.update(action, reward)

            avg_rewards[t] += reward
            pct_optimal_action[t] += action == testbed.get_optimal_action()

    avg_rewards /= num_runs
    pct_optimal_action /= num_runs
    pct_optimal_action *= 100.0

    return avg_rewards, pct_optimal_action


@app.route("/", methods=["GET"])
def form():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_once():
    # core params from dropdowns/inputs
    model = request.form.get("model", "epsilon-greedy")
    env = request.form.get("environment", "stationary")
    n_arms = to_int(request.form.get("n_arms"), 10)
    n_steps = to_int(request.form.get("n_steps"), 1000)
    mean = to_float(request.form.get("mean_q"), 0.0)
    std = to_float(request.form.get("std_q"), 1.0)

    # hyperparams
    params = {
        "epsilon": request.form.get("epsilon"),
        "alpha": request.form.get("alpha"),
        "initial_estimates": request.form.get("initial_q"),
        "ucb_c": request.form.get("ucb_c"),
    }

    avg_rewards, pct_optimal = run_simulation(
        model_key=model,
        env_key=env,
        num_arms=n_arms,
        num_steps=n_steps,
        mean_q=mean,
        std_q=std,
        params=params,
    )
    steps = list(range(1, n_steps + 1))

    return jsonify(
        {
            "ok": True,
            "steps": steps,
            "avg_rewards": avg_rewards.tolist(),
            "pct_optimal": pct_optimal.tolist(),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
