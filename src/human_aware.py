"""
Human-aware extension.

We add a randomized angular "keep-out" zone to the Pendulum task, standing in
for a human whose position changes from episode to episode. The robot (pendulum
tip) should still be controlled toward upright, but should avoid sweeping the
tip through the human's region.

Two things make this "human-aware and adaptive":
  1. The zone center is RANDOMIZED every episode, so a policy that simply
     memorized one path cannot succeed.
  2. The zone center is APPENDED to the observation, so the policy can SEE where
     the human is and adapt its behavior to the changing constraint.

Comparison story:
  - penalty = 0  -> "human-blind" baseline (never learns to avoid the zone)
  - penalty > 0  -> "human-aware" agent (learns to keep clear of the zone)
Both share the same observation space, so the comparison is clean.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def angle_diff(a, b):
    """Shortest signed angular distance between two angles (radians)."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


class HumanZoneWrapper(gym.Wrapper):
    def __init__(self, env, penalty=5.0, half_width=0.35):
        super().__init__(env)
        self.penalty = float(penalty)
        self.half_width = float(half_width)
        # Augment observation [cos th, sin th, th_dot] with the human location
        # encoded as [cos(zone), sin(zone)] -> 5-dim observation.
        low = np.concatenate([env.observation_space.low, [-1.0, -1.0]]).astype(np.float32)
        high = np.concatenate([env.observation_space.high, [1.0, 1.0]]).astype(np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.zone_center = 0.0

    def _sample_zone(self):
        # Place the human's keep-out zone close to the upright goal (theta = 0),
        # on a randomly chosen side. This creates real tension: a naive agent
        # balancing at upright will dip into the zone, while a human-aware agent
        # must bias its balance point to the clear side to stay out of it.
        sign = np.random.choice([-1.0, 1.0])
        return float(sign * np.random.uniform(0.3, 0.7))

    def _augment(self, obs):
        extra = [np.cos(self.zone_center), np.sin(self.zone_center)]
        return np.concatenate([obs, extra]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.zone_center = self._sample_zone()
        info["zone_center"] = self.zone_center
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        theta = np.arctan2(obs[1], obs[0])  # 0 = upright
        in_zone = abs(angle_diff(theta, self.zone_center)) < self.half_width
        info["in_zone"] = bool(in_zone)
        if in_zone:
            reward -= self.penalty
        return self._augment(obs), reward, terminated, truncated, info


def make_human_env(penalty=5.0, half_width=0.35):
    env = gym.make("Pendulum-v1")
    return HumanZoneWrapper(env, penalty=penalty, half_width=half_width)
