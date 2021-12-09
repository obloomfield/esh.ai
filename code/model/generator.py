import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, Activation, Dropout, UpSampling2D, Embedding
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.losses import binary_crossentropy

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # Hyperparameters
        
        self.depth = 64
        self.dim = 64 # i think this just needs to be 1/4 the final dim and everything works out
        self.dropout_rate = 0.4
        
        self.embed_size = 64
        
        self.embed = Sequential()
        
        self.embed.add(Embedding(59, self.embed_size))
        self.embed.add(Dense(self.embed_size))
        self.embed.add(LeakyReLU(alpha=0.03))
        
        # self.lstm = tf.keras.layers.LSTM(self.embed_size)
        
        # self.ca = Sequential()
        
        # self.ca.add(Dense())
        # self.ca.add(Dense())
        
        self.G = Sequential()

        # First stage
        self.G.add(Dense(self.dim*self.dim*self.depth))
        self.G.add(BatchNormalization(momentum=0.99))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((self.dim, self.dim, self.depth)))
        self.G.add(Dropout(self.dropout_rate))

        # Second stage
        self.G.add(UpSampling2D(size=2))
        self.G.add(Conv2DTranspose(self.depth // 2, 3, padding='SAME'))
        self.G.add(BatchNormalization(momentum=0.99))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D(size=2))
        self.G.add(Conv2DTranspose(self.depth // 4, 3, padding='SAME'))
        self.G.add(BatchNormalization(momentum=0.99))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D(size=2))
        self.G.add(Conv2DTranspose(self.depth // 8, 3, padding='SAME'))
        self.G.add(BatchNormalization(momentum=0.99))
        self.G.add(Activation('relu'))

        # Third stage: ouputs 256x256x1 image
        self.G.add(Conv2DTranspose(1, 3, padding='same'))
        self.G.add(Activation('tanh')) # maybe use softmax instead
        

    def call(self, x, z):
        
        embedding = self.embed(x)
        
        # debug: removing cond_augment for now
        # ca1 = Dense(self.dim * 2)(embedding)
        # mu, logvar = cond_aug(self, ca1)
        # z = tf.random.normal(self.depth*2, mean=mu, stdev=logvar)
        # glu = glu(embedding)
        
        x = tf.concat([embedding, z],axis=1)
        
        out = self.G(x) # runs the deconvolutions
        return out
    
    def loss(self, score):
        return tf.keras.losses.BinaryCrossentropy(tf.ones_like(score), score)