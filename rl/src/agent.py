import numpy as np

class Model(object):
    def __init__(self):
        pass
    def predict(self,s):
        pass
    def set_params(self,params):
        pass

class TabularQFunction(Model):
    def __init__(self, mu_init, std_init, num_actions, state_size):
        self.q = np.random.randn(state_size, num_actions)*std_init+mu_init

    def act(self, s):
        return np.argmax(self.q[s])

    def set_params(self, q):
        self.q = q.copy()

class PreprocessedTabularQFunction(Model):
    def __init__(self, mu_init, std_init, num_actions, preprocessor):
        self.preprocessor = preprocessor
        state_size = preprocessor.num_states
        self.q = np.random.randn(state_size, num_actions)*std_init+mu_init

    def act(self, s):
        sp = self.preprocessor.preprocess(s)
        return np.argmax(self.q[sp])

    def set_params(self, q):
        self.q = q.copy()

class NeuralNetworkQFunction(Model):
    def __init__(self, net):
        self.net = net
    def act(self,s):
        a = self.net.predict(s)
        return np.argmax(a)
    def set_params(self, params):
        self.net.set_params(params)
