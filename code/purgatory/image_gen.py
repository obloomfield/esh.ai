import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

class VQGAN(tf.keras.Model):
    def __init__(self):
        super(VQGAN, self).__init__()
        self.encoder = Sequential()
        self.decoder = Sequential()

    def call(self, inputs):
        pass