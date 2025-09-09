import numpy as np
from flask import Flask, jsonify, render_template, request

from n_armed_bandits.agents.epsilon_greedy import EpsilonGreedyAgent
from n_armed_bandits.agents.gradient import GradientBanditAgent
from n_armed_bandits.agents.ucb import UCBAgent
from n_armed_bandits.envs import NArmedTestbed, NonstationaryTestbed


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


# ----- factories (match your dropdowns) -----
def load_agent(model: str, n: int, params: dict):
    if model == "Epsilon Greedy":
        return EpsilonGreedyAgent(
            n=n,
            epsilon=to_float(params.get("epsilon"), 0.1),
            alpha=to_float(params.get("alpha")),  # None => sample-average updates
            initial_estimates=to_float(params.get("initial_estimates"), 0.0),
        )
    elif model == "Upper Confidence Bound (UCB)":
        return UCBAgent(
            n=n,
            c=to_float(params.get("c"), 2.0),
            alpha=to_float(params.get("alpha")),  # optional constant-α
        )
    elif model == "Gradient Bandit":
        return GradientBanditAgent(
            n=n,
            alpha=to_float(params.get("alpha"), 0.1),
            use_baseline=to_bool(params.get("use_baseline"), True),
        )
    raise ValueError(f"Unknown model: {model}")


def load_env(env: str, n: int, mean: float, std: float):
    if env == "Stationary":
        return NArmedTestbed(n=n, mean=mean, std=std)
    elif env == "Nonstationary":
        return NonstationaryTestbed(n=n, mean=mean, std=std)
    raise ValueError(f"Unknown environment: {env}")


@app.route("/", methods=["GET"])
def form():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_once():
    # core params from dropdowns/inputs
    model = request.form.get("model", "Epsilon Greedy")
    env_name = request.form.get("env", "Stationary")
    n_arms = to_int(request.form.get("n_arms"), 10)
    n_steps = to_int(request.form.get("n_steps"), 1000)
    mean = to_float(request.form.get("mean"), 0.0)
    std = to_float(request.form.get("std"), 1.0)

    # hyperparams
    params = {
        "epsilon": request.form.get("epsilon"),
        "alpha": request.form.get("alpha"),
        "initial_estimates": request.form.get("initial_estimates"),
        "c": request.form.get("c"),
        "use_baseline": request.form.get("use_baseline"),
    }

    # build env + agent
    env = load_env(env_name, n_arms, mean, std)
    agent = load_agent(model, n_arms, params)

    # simulate
    actions, rewards = [], []
    optimal_hits = 0

    for _ in range(n_steps):
        a = int(agent.select_action())
        r = float(env.get_reward(a))
        agent.update(a, r)

        actions.append(a)
        rewards.append(r)

        # track optimal % if env provides argmax of true q*
        try:
            if a == int(env.get_optimal_action()):
                optimal_hits += 1
        except Exception:
            pass

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    optimal_pct = float(optimal_hits / n_steps) if n_steps else 0.0

    # expose internal learning state for teaching UI
    # (Epsilon/UCBAgent: q; GradientBandit: h preferences, plus pi if you want)
    q_est = None
    for attr in ("q", "h", "Q", "H", "preferences"):
        if hasattr(agent, attr):
            arr = getattr(agent, attr)
            try:
                q_est = [float(x) for x in arr]
            except Exception:
                pass
            break

    pi = getattr(agent, "pi", None)
    if isinstance(pi, np.ndarray):
        pi = pi.tolist()

    # true means (JSON-friendly)
    q_star = getattr(env, "q_star", None)
    if isinstance(q_star, np.ndarray):
        q_star = q_star.tolist()

    return jsonify(
        {
            "ok": True,
            "model": model,
            "env": env_name,
            "n_arms": n_arms,
            "n_steps": n_steps,
            "avg_reward": avg_reward,
            "optimal_pct": optimal_pct,
            "actions": actions,
            "rewards": rewards,
            "q_est": q_est,  # agent estimates or preferences
            "pi": pi,  # gradient bandit policy (if applicable)
            "q_star": q_star,  # true action values
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
