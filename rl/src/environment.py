import importlib
import gym

def get_environment(params):
    if params['ENV_PROVIDER'] == 'GYM':
        return get_gym_environment(params)

def get_gym_environment(params):
    env = gym.make(params['ENV_NAME'])
    if not "ENV_PARAMS" in params:
        return env
    else:
        return env.env.__class__(**params['ENV_PARAMS'])
