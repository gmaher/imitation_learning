import numpy as np
from imitation_learning import loss
import tensorflow as tf

class DiscreteActorCritic:
    def __init__(self, policy, value_function, preprocess, learning_rate):
        self.policy = policy
        self.vf  = value_function
        self.pre = preprocess

        self.opt = tf.optimizers.SGD(learning_rate=learning_rate,
            momentum=0.9)

    def act(self, s):
        S = self.pre(s)
        p = self.policy.predict(S)[0]
        z = np.random.rand()

        return np.random.choice(len(p), p=p)

    def train(self, S, A, R, I):
        values = self.vf(S)

        gp = loss.policy_gradient(self.policy, values, S, A, R)

        self.opt.apply_gradients(zip(gp,self.policy.trainable_variables))

        gv = loss.value_gradient(self.vf, S, R, I)

        self.opt.apply_gradients(zip(gv,self.vf.trainable_variables))

        l = loss.discrete_policy_loss(self.policy(S),values,A,R)

        return l
