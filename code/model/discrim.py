import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, Dropout, LeakyReLU, Embedding
from tensorflow.keras.losses import BinaryCrossentropy

class Discriminator(tf.keras.Model):
            
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
        self.embed_size = 64
        
        self.embed = Sequential()
        
        self.embed.add(Embedding(self.embed_size))
        self.embed.add(LeakyReLU(alpha=0.03))
        
        self.D = Sequential()
        
        depth = 64
        dropout = 0.4
        input_shape = (256, 256, 3)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
        padding='same', activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same', activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))

        
        
    @tf.function
    def call(self, latent, embedding):
        x = tf.concat([latent,embedding],axis=1)
        out = self.D(x)
        return out
    
    def score(self, score):
        return BinaryCrossentropy(tf.ones_like(score), score)

    def loss(self, pred, labels):
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pred), logits=pred))
        D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(labels), logits=labels))
        return D_loss
    