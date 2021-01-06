#!/usr/bin/env python

import cv2
import gym_super_mario_bros
import neat
import numpy as np
from gym.envs.classic_control import rendering
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

viewer = rendering.SimpleImageViewer()
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

genome_ids_to_watch = [79529]


def nnout_to_action(nnout):
    return nnout.index(max(nnout))


def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        if not err:
            print(
                "Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(
                    k, l
                )
            )
            err.append("logged")
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


def eval_genomes(genomes, config):

    # loop through each genome in the genomes population (created in the next cell)
    for genome_id, genome in genomes:
        # if genome_id not in genome_ids_to_watch:
        #     continue
        state = env.reset()

        iny, inx, inc = env.observation_space.shape
        inx = int(inx / 8)
        iny = int(iny / 8)

        net = neat.nn.FeedForwardNetwork.create(genome, config)

        done = False
        fitness_current = 0.0
        frames = 0
        old_x = 0

        while not done:
            state = cv2.resize(state, (inx, iny))
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            state = np.reshape(state, (inx, iny))
            rgb = env.render("rgb_array")
            upscaled = repeat_upsample(rgb, 4, 4)
            viewer.imshow(upscaled)

            imgarray = np.ndarray.flatten(state)
            nnout = net.activate(imgarray)
            action = nnout_to_action(nnout)
            state, rew, done, info = env.step(action)
            fitness_current += rew

            frames += 1
            if frames % 50 == 0:
                if old_x == info["x_pos"]:
                    done = True
                else:
                    old_x = info["x_pos"]

        print(f"GenomeID: {genome_id}, Fitness: {fitness_current}")
        genome.fitness = fitness_current


config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config-feedforward",
)

p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-1157")

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.run(eval_genomes)
