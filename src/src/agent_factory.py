from src.agent import TabularPolicy

import numpy as np

def get_agent(agent_config, env_config):
    if agent_config['TYPE'] == 'TabularPolicy':
        return get_tabular_policy(agent_config, env_config)
    else:
        raise RuntimeError("Unrecognized agent")

def get_tabular_policy(agent_config, env_config):
    return TabularPolicy(state_size=env_config['STATE_SIZE'][0],
        num_actions=env_config['NUM_ACTIONS'])
