import numpy as np
import tensorflow as tf
from glu import FastGLU

class CA(tf.keras.Model):
            
    def __init__(self, input_size, conditional_size , **kwargs):
        super().__init__(**kwargs)
        self.text_embed_dim = input_size
        self.cond_dim = conditional_size
        
        self.dense = tf.keras.layers.Dense(self.text_embed_dim, self.cond_dim * 4, use_bias=True)
        self.relu = FastGLU()
    
    def encode(self, t_embed):
        return 0
        
    # using the reparam technique from the VAE project:
    def reparam(self, mu, logvar):
        sigma = tf.math.sqrt(tf.math.exp(logvar))
        
        epsilon = tf.random.normal(mu.shape, mean=0, stddev=1)
        z = sigma * epsilon + mu

        return z
    
    @tf.function
    def call(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar