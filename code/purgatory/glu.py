import numpy as np
import tensorflow as tf

class FastGLU(tf.keras.Model):
            
    def __init__(self, input_size, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.linear = tf.keras.layers.Dense(input_size, input_size**2)
        
    @tf.function
    def call(self, x):
        out = self.linear(x)
        return out[:,:self.input_size] * tf.math.sigmoid(out[:,self.input_size:])