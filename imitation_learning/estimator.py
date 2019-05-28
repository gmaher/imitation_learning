import tensorflow as tf
from tensorflow import keras

class LinearModel():
    def __init__(self, input_size, output_size, activation):
        self.input_size  = input_size
        self.output_size = output_size

        self.model = keras.Sequential([
            keras.layers.Input(shape=[self.input_size]),
            keras.layers.Dense(self.output_size, activation=activation)
        ])

    def predict(self, s):
        return self.model.predict(s)

    def __call__(self, s):
        return self.model(s)
