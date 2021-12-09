import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.losses import binary_crossentropy

class FastGLU(tf.keras.Model):
            
    def __init__(self, rnn_size,  **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        #FROM CONTROLGAN
        # stride 1,1 relu
        # stride 2,2 relu
        # stride 2,2, relu
        # 2,2 tanh
        
        
        
    @tf.function
    def call(self, latent, embedding):
        x = tf.concat([latent,embedding],axis=1)
        out = self.deconvolutional(x)
        return out
    
    def score(self, score):
        return tf.keras.losses.BinaryCrossentropy(tf.ones_like(score), score)

    def loss(self, )
    