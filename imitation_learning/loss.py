import numpy as np
import tensorflow as tf

def discrete_policy_loss(p, v, a, r):
    """
    p - array (n_episodes x n_steps x num_actions)
    v - array (n_episodes x n_steps)
    a - array (n_episodes x n_steps x num_actions) - binary array
    r - array (n_episodes x n_steps)
    """
    log_pr = -tf.math.log(p)*a*(r - v)

    entropy = -tf.reduce_mean(p*tf.math.log(p))

    return tf.reduce_sum(log_pr)/a.shape[0] + 0.01*entropy

def policy_gradient(model, values, s, a, r):
    with tf.GradientTape() as t:
        loss = discrete_policy_loss(model(s), values, a, r)
    return t.gradient(loss, model.trainable_variables)

def value_gradient(model, s, r, I):
    with tf.GradientTape() as t:
        loss = tf.reduce_mean(tf.square(model(s)-r)*I)
    return t.gradient(loss, model.trainable_variables)
