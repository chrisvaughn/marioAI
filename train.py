import argparse
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


class TrainAndLoggingCallback(BaseCallback):
    def __init__(
        self,
        check_freq,
        save_path,
        resume_training_interations=0,
        verbose=1,
    ):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.resume_training_interations = resume_training_interations

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            total_interations = self.resume_training_interations + self.n_calls
            model_path = os.path.join(self.save_path, f"best_model_{total_interations}")
            self.model.save(model_path)

        return True


def main(model_path=None, resume_training_interations=0):
    env = gym_super_mario_bros.make("SuperMarioBros-v0", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order="last")

    callback = TrainAndLoggingCallback(
        check_freq=10000,
        save_path=CHECKPOINT_DIR,
        resume_training_interations=resume_training_interations,
    )
    if model_path is not None:
        model = PPO.load(
            model_path,
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=0.000001,
            n_steps=512,
        )
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=0.000001,
            n_steps=512,
        )
    model.learn(total_timesteps=1000000, callback=callback)
    model.save("thisisatestmodel")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--resume-model", help="path to model to resume training")
    argparse.add_argument(
        "--resume-interations",
        type=int,
        default=None,
        help="number of interations to resume training from",
    )
    args = argparse.parse_args()
    if args.resume_model:
        if args.resume_interations is None:
            raise ValueError(
                "Must provide number of interations to resume training from"
            )
        main(args.resume_model, args.resume_interations)
    else:
        main()
