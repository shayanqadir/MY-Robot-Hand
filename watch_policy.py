"""
Watch a trained (or checkpointed) policy control the hand in the viewer.

Run: python watch_policy.py hand_crawl_ppo_final.zip
  or: python watch_policy.py checkpoints/hand_crawl_ppo_500000_steps.zip

If vecnormalize_stats.pkl exists (saved automatically at the end of
train_ppo.py), it's loaded and applied - the policy was trained on
*normalized* observations, so replaying it on raw observations without this
will make it behave incorrectly, even though no error is raised.
"""

import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from hand_env import HandCrawlEnv

VECNORMALIZE_PATH = "vecnormalize_stats.pkl"


def main():
    if len(sys.argv) < 2:
        print("Usage: python watch_policy.py <path_to_model.zip>")
        sys.exit(1)

    model_path = sys.argv[1]
    policy = PPO.load(model_path)

    env = HandCrawlEnv(render_mode="human")

    vecnorm = None
    if os.path.exists(VECNORMALIZE_PATH):
        dummy = DummyVecEnv([lambda: HandCrawlEnv()])
        vecnorm = VecNormalize.load(VECNORMALIZE_PATH, dummy)
        vecnorm.training = False   # don't keep updating stats at inference time
        vecnorm.norm_reward = False
        print(f"Loaded observation normalization from {VECNORMALIZE_PATH}")
    else:
        print("No vecnormalize_stats.pkl found - proceeding without normalization "
              "(only correct if you trained without VecNormalize).")

    obs, _ = env.reset()

    while True:
        obs_in = vecnorm.normalize_obs(obs) if vecnorm else obs
        action, _ = policy.predict(obs_in, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(env.model.opt.timestep * 5)  # keep pace with FRAME_SKIP roughly real-time

        if terminated or truncated:
            print(f"Episode ended (x_position={info['x_position']:.3f})")
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
