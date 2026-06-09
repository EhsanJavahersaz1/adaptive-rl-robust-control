"""
Compare two agents on the human-aware task by how much time the pendulum tip
spends inside the randomized human keep-out zone (lower = more human-aware).

    python src/evaluate_human_aware.py \
        --models models/human_blind models/human_aware \
        --labels "Human-blind" "Human-aware" \
        --out results/human_avoidance.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from human_aware import make_human_env


def eval_agent(model, episodes=50):
    env = make_human_env(penalty=5.0)  # penalty here only affects the logged
    in_zone_fracs, returns = [], []     # return, not the (deterministic) behavior
    for _ in range(episodes):
        obs, _ = env.reset()
        done = trunc = False
        steps = inzone = 0
        total = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            total += r
            steps += 1
            inzone += int(info["in_zone"])
        in_zone_fracs.append(inzone / steps)
        returns.append(total)
    env.close()
    return 100.0 * np.mean(in_zone_fracs), np.mean(returns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--out", type=str, default="results/human_avoidance.png")
    args = p.parse_args()

    labels, zone_pct = [], []
    for path, label in zip(args.models, args.labels):
        model = SAC.load(path)
        pct, ret = eval_agent(model, args.episodes)
        print(f"{label:>14}: {pct:5.1f}% time in human zone | mean return {ret:8.1f}")
        labels.append(label)
        zone_pct.append(pct)

    plt.figure(figsize=(5.5, 4))
    bars = plt.bar(labels, zone_pct, color=["#c0504d", "#4f81bd"])
    for b, v in zip(bars, zone_pct):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center")
    plt.ylabel("Time inside human keep-out zone (%)")
    plt.title("Human-aware avoidance (lower is better)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
