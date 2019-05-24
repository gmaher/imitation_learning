import numpy as np

class Model(object):
    def __init__(self):
        pass
    def predict(self,s):
        pass

class TabularPolicy(Model):
    def __init__(self, num_states, num_actions):
        init = 1.0/num_actions
        self.num_actions = num_actions
        self.num_states  = num_states
        self.params = np.zeros((num_states,num_actions))+init

    def predict(self,s):
        return np.random.choice(self.num_actions,
            p=self.params[s])
