#!/usr/bin/env python

import pickle

import cv2
import gym_super_mario_bros
import neat
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def nnout_to_action(nnout):
    return nnout.index(max(nnout))


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, genome_id)


def eval_genome(genome, config, genome_id=None):
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    state = env.reset()

    iny, inx, inc = env.observation_space.shape
    inx = int(inx / 8)
    iny = int(iny / 8)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    done = False
    fitness_current = 0.0
    frames = 0
    old_x = 0
    lives_remaining = 2
    total_coins = 0
    current_status = "small"

    while not done:
        state = cv2.resize(state, (inx, iny))
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = np.reshape(state, (inx, iny))
        # env.render()

        imgarray = np.ndarray.flatten(state)
        nnout = net.activate(imgarray)
        action = nnout_to_action(nnout)
        state, rew, done, info = env.step(action)
        fitness_current += rew

        # if mario gets to the flag give a very high reward meet fitness_threshold
        if info["flag_get"]:
            fitness_current += 500000

        # extra penalty for dying
        if info["life"] < lives_remaining:
            lives_remaining = info["life"]
            fitness_current -= 250

        # bonus for managing to change status
        if current_status != info["status"]:
            current_status = info["status"]
            if info["status"] != "small":
                fitness_current += 100

        total_coins = info["coins"]
        frames += 1
        if frames % 50 == 0:
            if old_x == info["x_pos"]:
                done = True
            else:
                old_x = info["x_pos"]

    # bonus for collecting coins
    fitness_current += total_coins * 10

    if genome_id:
        print(f"GenomeID: {genome_id}, Fitness: {fitness_current}")
    else:
        print(f"Fitness: {fitness_current}")

    env.close()
    return fitness_current


config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config-feedforward",
)
p = neat.Population(config)
# p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-144")

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()

p.add_reporter(stats)

# Save the process after each 10 frames
p.add_reporter(neat.Checkpointer(1))

pe = neat.ParallelEvaluator(10, eval_genome)
winner = p.run(pe.evaluate)

# winner = p.run(eval_genomes)

with open("winner.pkl", "wb") as output:
    pickle.dump(winner, output, 1)
