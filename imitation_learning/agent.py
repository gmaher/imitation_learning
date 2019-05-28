import numpy as np
import tensorflow as tf
from tensorflow import keras

def do_nothing(x):
    return x

class Agent(object):
    def __init__(self, model, preprocess_state=do_nothing):
        self.model = model
        self.preprocess_state = preprocess_state
    def act(self,s):
        pass

class DiscreteActionAgent(Agent):
    def act(self, s):
        S = self.preprocess_state(s)

        p = self.model.predict(S)[0]

        return np.random.choice(len(p), p=p)

class ContinuousActionAgent(Agent):
    def act(self, s):
        pass
