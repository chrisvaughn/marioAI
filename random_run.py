#!/usr/bin/env python

import cv2
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state = env.reset()
iny, inx, inc = env.observation_space.shape
inx = int(inx / 8)
iny = int(iny / 8)

done = False
for step in range(5000):
    if done:
        state = env.reset()
    state = cv2.resize(state, (inx, iny))
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = np.reshape(state, (inx, iny))

    imgarray = np.ndarray.flatten(state)
    print(len(imgarray))

    action = env.action_space.sample()
    print(action)
    state, reward, done, info = env.step(action)

    env.render()

env.close()
