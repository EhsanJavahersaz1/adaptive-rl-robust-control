"""
Train a SAC agent on Pendulum-v1.

Two training regimes (this contrast is the core experiment):
  --mode nominal     : train only on the default physics (fixed distribution)
  --mode randomize   : train with domain randomization (robust agent)

Example:
    python src/train.py --mode nominal   --steps 60000 --out models/sac_nominal
    python src/train.py --mode randomize --steps 60000 --out models/sac_robust
"""

import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from envs import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["nominal", "randomize"], default="nominal")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="models/sac_agent")
    args = p.parse_args()

    env = Monitor(make_env(mode=args.mode))
    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.98,
        tau=0.02,
        train_freq=1,
    )
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip  (trained with mode='{args.mode}')")


if __name__ == "__main__":
    main()
