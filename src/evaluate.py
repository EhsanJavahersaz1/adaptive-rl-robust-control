"""
Evaluate trained agents under a sweep of dynamics shift and plot the result.

We take each saved agent and test it on Pendulum environments whose mass is
scaled away from the nominal value it was (mostly) trained on. A robust agent
should degrade more gracefully than one trained on the fixed nominal physics.

Example:
    python src/evaluate.py --models models/sac_nominal models/sac_robust \
                           --labels "Nominal training" "Domain randomization"
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from envs import make_env, NOMINAL


def eval_one(model, mass, episodes=20):
    """Mean episode return for a given pendulum mass."""
    env = make_env(mode="fixed", fixed={"m": mass})
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = trunc = False
        total = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(action)
            total += r
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", type=str, default="results/shift_robustness.png")
    args = p.parse_args()
    assert len(args.models) == len(args.labels)

    # Sweep pendulum mass from light to heavy (nominal m = 1.0).
    masses = np.linspace(0.4, 2.2, 10)

    plt.figure(figsize=(7, 4.5))
    for path, label in zip(args.models, args.labels):
        model = SAC.load(path)
        means, stds = [], []
        for m in masses:
            mu, sd = eval_one(model, m, args.episodes)
            means.append(mu)
            stds.append(sd)
        means, stds = np.array(means), np.array(stds)
        plt.plot(masses, means, marker="o", label=label)
        plt.fill_between(masses, means - stds, means + stds, alpha=0.15)

    plt.axvline(NOMINAL["m"], ls="--", c="gray", lw=1, label="nominal mass")
    plt.xlabel("Pendulum mass (nominal = 1.0)")
    plt.ylabel("Mean episode return (higher is better)")
    plt.title("Policy robustness under dynamics shift")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
