from src.agent import TabularQFunction, PreprocessedTabularQFunction
from src.agent import NeuralNetworkQFunction
from src.preprocessor import TablePreprocessor

import numpy as np

def get_agent(agent_config, env_config):
    if agent_config['TYPE'] == 'TabularQFunction':
        return get_tabular_q(agent_config, env_config)
    elif agent_config['TYPE'] == 'PreprocessedTabularQFunction':
        return get_preprocessed_q(agent_config, env_config)
    else:
        raise RuntimeError("Unrecognized agent")

def get_tabular_q(agent_config, env_config):
    return TabularQFunction(state_size=env_config['STATE_SIZE'][0],
        num_actions=env_config['NUM_ACTIONS'],
        mu_init=agent_config['Q_INIT'],
        std_init=agent_config['Q_STD'])

def get_preprocessed_q(agent_config, env_config):
    ranges = np.array(env_config['STATE_RANGES'])
    bins   = agent_config['BINS']

    preprocessor = TablePreprocessor(ranges, bins)

    agent = PreprocessedTabularQFunction(
        num_actions=env_config['NUM_ACTIONS'],
        mu_init=agent_config['Q_INIT'],
        std_init=agent_config['Q_STD'],
        preprocessor=preprocessor)

    return agent
