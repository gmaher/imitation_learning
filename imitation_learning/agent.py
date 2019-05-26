import numpy as np
import tensorflow as tf
from tensorflow import keras

class Agent(object):
    def __init__(self, model):
        self.model = model
    def act(self,s):
        pass

class DiscreteActionAgent(Agent):
    def act(self, s):
        sp = s
        if (len(s.shape)==1):
            sp = s[np.newaxis,:]

        p = self.model.predict(sp)[0]

        return np.random.choice(len(p), p=p)

class ContinuousActionAgent(Agent):
    def act(self, s):
        pass
