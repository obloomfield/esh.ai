import numpy as np
import tensorflow as tf

class TextEncode(tf.keras.Model):
            
    def __init__(self, input_size, embedding_size , beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = input_size
        self.num_embeddings = embedding_size
        self.beta = beta
        
        uniform = tf.random_uniform_initializer()
        self.embedding = tf.Variable(
            initial_value=uniform(shape=(self.embed_dim, self.num_embeddings), dtype="float32"),
            trainable=True
            )