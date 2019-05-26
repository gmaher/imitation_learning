from imitation_learning.loss import discrete_policy_loss, continuous_policy_loss

class Model(object):
    def __init__(self, input_size, output_size, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.setup()
        self.finalize()
    def setup(self):
        pass
    def finalize(self):
        self.model.compile(optimizer='adam',
            loss=self.loss,
            learning_rate=learning_rate)

    def predict(self, s):
        pass
    def train(self, T):
        pass

class LinearClassifier(Model):
    def setup(self):
        self.model = keras.Sequential([
            keras.layers.Input(shape=(self.input_size)),
            keras.layers.Dense(self.output_size, activation=tf.nn.softmax)
        ])

        self.loss = discrete_policy_loss
