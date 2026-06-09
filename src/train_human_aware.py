"""
Train a SAC agent on the human-aware Pendulum task.

    # human-blind baseline (no penalty -> never learns to avoid the human)
    python src/train_human_aware.py --penalty 0 --steps 80000 --out models/human_blind

    # human-aware agent (penalized for entering the human's zone)
    python src/train_human_aware.py --penalty 5 --steps 80000 --out models/human_aware
"""

import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from human_aware import make_human_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--penalty", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="models/human_aware")
    args = p.parse_args()

    env = Monitor(make_human_env(penalty=args.penalty))
    model = SAC("MlpPolicy", env, seed=args.seed, verbose=1,
                learning_rate=1e-3, buffer_size=100_000, batch_size=256,
                gamma=0.98, tau=0.02, train_freq=1)
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(args.out)
    print(f"Saved {args.out}.zip  (penalty={args.penalty})")


if __name__ == "__main__":
    main()
