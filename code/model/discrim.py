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
        
        self.embed.add(Embedding(59, self.embed_size))
        # self.embed.add(Dense(self.embed_size))
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
        self.D.add(Flatten())
        
        self.post_convolution = Sequential()
        self.post_convolution.add(Dense(1))
        self.post_convolution.add(Activation('sigmoid'))

    def call(self, image, labels):
        embed = self.embed(labels)
        x = self.D(image)
        
        x = tf.concat([x,embed],axis=1)
        out = self.post_convolution(x)
        return out
    
    def score(self, score):
        return tf.keras.losses.BinaryCrossentropy(tf.ones_like(score), score)

    def loss(self, fake_trick_score, all_real_score, rand_label_score): # extension on loss from lab:
        
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(all_real_score), logits=all_real_score))
        D_loss += (1/2) * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_trick_score), logits=fake_trick_score))
        D_loss += (1/2) * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(rand_label_score), logits=rand_label_score))
        return D_loss
