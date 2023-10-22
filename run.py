import os

import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"

# monkey patch reset method for incompatibility between nes-py 8.2.1 (and earlier) and gym 0.26.0 (and later).
_reset = JoypadSpace.reset


def reset(self, *args, **kwargs):
    obs_info = _reset(self)
    obs, info = obs_info if type(obs_info) == tuple else (obs_info, {})
    return obs, info


JoypadSpace.reset = reset


env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

model = PPO.load("./train/best_model_40000.zip")

done = True
state = env.reset()
# Loop through each frame in the game
for step in range(100000):
    # Start the game to begin with
    if done:
        # Start the gamee
        env.reset()
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
# Close the game
env.close()
