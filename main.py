import argparse

import sys
import os
sys.path.append(os.path.abspath('..'))
import numpy as np

import config
from imitation_learning.util import read_json
from imitation_learning.environment import get_environment
from imitation_learning.agent_factory import get_agent
from imitation_learning.simulator import Simulator

parser = argparse.ArgumentParser()
parser.add_argument('-agent', type=str)
parser.add_argument('-env', type=str)
args = parser.parse_args()

#######################################
# Set up simulation
#######################################
agent_cfg   = read_json(args.agent)
env_cfg     = read_json(args.env)

agent        = get_agent(agent_cfg, env_cfg)
env          = get_environment(env_cfg)

sim   = Simulator(env, agent, discount=1.0)

rewards = []
avg_r   = 0.0
for i in range(config.num_episodes):
    R = sim.run(render=config.render, num_steps=config.num_steps)
    avg_r = config.alpha*avg_r + (1.0-config.alpha)*R
    rewards.append(avg_r)
    print(R, avg_r)

    replay.append(sim.tuples)

    T = replay.sample()

    model.train(T)

import matplotlib.pyplot as plt
plt.plot(rewards)
plt.show()
