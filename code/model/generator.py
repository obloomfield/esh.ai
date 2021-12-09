import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.losses import binary_crossentropy

class Generator(tf.keras.Model):
            
    def __init__(self, rnn_size,  **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
       
        self.deconvolutional = Sequential(
            Dense(8*8*256)
            Reshape(8,8,256) #know its 8x8 so grow it
            conv2d(128)
            conv2d(64)
            conv2d(32, )
            #maybe batch normalize?
            conv2d(3, [5,5],) #3 color channels brackets is filter size, all 5,5
            
        )
        # stride 1,1 relu
        # stride 2,2 relu
        # stride 2,2, relu
        # 2,2 tanh
        
        
        
    @tf.function
    def call(self, latent, embedding):
        x = tf.concat([latent,embedding],axis=1)
        out = self.deconvolutional(x)
        return out
    
    def loss(self, score):
        return tf.keras.losses.BinaryCrossentropy(tf.ones_like(score), score)