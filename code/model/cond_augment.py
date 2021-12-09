import numpy as np
import tensorflow as tf
from code.purgatory.glu import FastGLU

class CA():

    # implementation of conditional augmentation, 
    # kind of like the normal dist. add to our VAE project: 

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def cond_aug(model, x):
        mu, logvar = model.encode(x)
        z = model.reparametrize(mu, logvar)
        return z, mu, logvar
    
    def glu(model, x):
        size = x.shape(1) / 2 
        # divide into two channels
        return x[:,:size] * tf.math.sigmoid(x[:,size:])
        
    
    def __init__(self, input_size, conditional_size , **kwargs):
        super().__init__(**kwargs)
        self.text_embed_dim = input_size
        self.cond_dim = conditional_size
        
        self.dense = tf.keras.layers.Dense(self.text_embed_dim, self.cond_dim * 4, use_bias=True)

    
    
        
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