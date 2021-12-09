import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, Activation, Dropout, UpSampling2D, Embedding
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.losses import binary_crossentropy

class Discriminator(tf.keras.Model):
            
    def __init__(self,  **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.depth = 64
        self.dim = 64 # i think this just needs to be 1/4 the final dim and everything works out
        self.dropout_rate
        
        self.embed_size = 64
        
        
        self.embed = Sequential()
        
        self.embed.add(Embedding(self.embed_size))
        self.embed.add(LeakyReLU(alpha=0.03))
        
        self.ca = Sequential()
        
        self.ca.add(Dense())
        
        self.G = Sequential()

        # First stage
        self.G.add(Dense(self.dim*self.dim*self.depth))
        self.G.add(BatchNormalization(momentum=0.99))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((self.dim, self.dim, self.depth)))
        self.G.add(Dropout(self.dropout_rate))

        # Second stage
        self.G.add(UpSampling2D(size=2))
        self.G.add(Conv2DTranspose(self.depth // 2, 5, padding='SAME'))
        self.G.add(BatchNormalization(momentum=0.99))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D(size=2))
        self.G.add(Conv2DTranspose(self.depth // 4, 5, padding='SAME'))
        self.G.add(BatchNormalization(momentum=0.99))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D(size=2))
        self.G.add(Conv2DTranspose(self.depth // 8, 5, padding='SAME'))
        self.G.add(BatchNormalization(momentum=0.99))
        self.G.add(Activation('relu'))

        # Third stage: ouputs 264x264x1 image
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('tanh')) # maybe use softmax instead
        
        
        
    @tf.function
    def call(self, latent, embedding):
        x = tf.concat([latent,embedding],axis=1)
        out = self.deconvolutional(x)
        return out
    
    def score(self, score):
        return tf.keras.losses.BinaryCrossentropy(tf.ones_like(score), score)

    def loss(self, )
    