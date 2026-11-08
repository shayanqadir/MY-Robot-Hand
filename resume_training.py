"""
Resume training from the latest checkpoint, with N_ENVS lowered to match
this machine's 2 physical cores (was 8 - badly oversubscribed a 2-core CPU).

Run: python resume_training.py
"""

import os
import re
import glob

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from hand_env import HandCrawlEnv

gym.register(id="HandCrawl-v0", entry_point=HandCrawlEnv)

CHECKPOINT_DIR = "./checkpoints"
N_ENVS = 2                 # was 8 - this machine has 2 physical cores
TOTAL_TIMESTEPS = 10_000_000   # same overall target as the original run
CHECKPOINT_EVERY = 50_000
RAND_START_SCALE = 0.0
RAND_RAMP_FRACTION = 0.7
RAND_UPDATE_EVERY = 20_000


class RandomizationCurriculum(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total = total_timesteps
        self._last_update = -RAND_UPDATE_EVERY

    def _scale(self):
        progress = self.num_timesteps / (self.total * RAND_RAMP_FRACTION)
        return min(1.0, RAND_START_SCALE + (1.0 - RAND_START_SCALE) * progress)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= RAND_UPDATE_EVERY:
            s = self._scale()
            self.training_env.env_method("set_randomization_scale", s)
            self.logger.record("rand/scale", s)
            self._last_update = self.num_timesteps
        return True


def find_latest_checkpoint():
    zips = glob.glob(os.path.join(CHECKPOINT_DIR, "hand_crawl_ppo_*_steps.zip"))
    if not zips:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    def steps_of(path):
        m = re.search(r"_(\d+)_steps\.zip$", path)
        return int(m.group(1)) if m else -1
    latest = max(zips, key=steps_of)
    steps = steps_of(latest)
    vecnorm = latest.replace(f"_{steps}_steps.zip", f"_vecnormalize_{steps}_steps.pkl")
    return latest, vecnorm, steps


def main():
    model_path, vecnorm_path, current_steps = find_latest_checkpoint()
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"Missing matching normalization file: {vecnorm_path}")
    print(f"Resuming from {model_path} (currently at {current_steps:,} steps)")
    print(f"Using N_ENVS={N_ENVS} (matches this CPU's 2 physical cores)")

    env = make_vec_env("HandCrawl-v0", n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    env = VecNormalize.load(vecnorm_path, env)
    env.training = True
    env.norm_reward = True

    model = PPO.load(model_path, env=env, device="cpu")
    # keep the curriculum consistent with progress so far
    model.env.env_method("set_randomization_scale",
                          min(1.0, current_steps / (TOTAL_TIMESTEPS * RAND_RAMP_FRACTION)))

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_EVERY // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="hand_crawl_ppo",
        save_vecnormalize=True,
    )
    curriculum = RandomizationCurriculum(TOTAL_TIMESTEPS)
    curriculum._last_update = current_steps  # don't immediately re-log a stale scale

    remaining = TOTAL_TIMESTEPS - current_steps
    if remaining <= 0:
        print("Already at or past the target timestep count - nothing to resume.")
        return

    model.learn(total_timesteps=remaining,
                callback=[checkpoint_callback, curriculum],
                reset_num_timesteps=False)

    model.save("hand_crawl_ppo_final")
    env.save("vecnormalize_stats.pkl")
    print("Training done. Saved hand_crawl_ppo_final.zip and vecnormalize_stats.pkl")


if __name__ == "__main__":
    main()
