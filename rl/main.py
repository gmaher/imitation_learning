import argparse

import sys
import os
sys.path.append(os.path.abspath('..'))
import numpy as np

import config
from src.util import read_json
from src.environment import get_environment
from src.agent_factory import get_agent
from src.simulator import Simulator
from src.distribution_factory import get_distribution
from ce.ce import GeneralizedCrossEntropyOptimizer

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
args = parser.parse_args()

#######################################
# Set up simulation
#######################################
cfg   = read_json(args.config)

env          = get_environment(cfg)
agent        = get_agent(cfg)
distribution = get_distribution(cfg)

sim   = Simulator(env, agent, discount=1.0)

def step(theta):
    agent.set_params(theta)
    return sim.run()

cross = GeneralizedCrossEntropyOptimizer(
    f=step,
    distribution=distribution,
    N=config.Nr,
    Ne=config.Ne,
    gam0=config.gam0)

rewards = []
avg_r   = 0.0
for i in range(config.num_episodes):
    cross.step()
    R = sim.run(render=config.render, num_steps=config.num_steps)
    avg_r = config.alpha*avg_r + (1.0-config.alpha)*R
    rewards.append(avg_r)
    print(R, avg_r)

import matplotlib.pyplot as plt
plt.plot(rewards)
plt.show()
