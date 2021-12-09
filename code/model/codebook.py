import numpy as np
import tensorflow as tf

class Codebook(tf.keras.Model):
    # OUR IMPLEMENTATION OF A VECTOR QUANTIZER
    
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

        
    
    @tf.function
    def call(self, x):
        
        input_shape = tf.shape(x)
        flat = tf.reshape(x,[-1, self.embedding_dim])
        
        entries = self.get_entries(flat)
        one_hot = tf.one_hot(entries, self.num_embeddings)
        quantized = tf.matmul(one_hot, self.embedding, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)
        
        #LOSS OF CODEBOOK 
        commit_loss = self.beta * tf.reduce_mean(tf.stop_gradient(quantized) - x) ** 2
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commit_loss + codebook_loss)
        
        estimator = x + tf.stop_gradient(quantized - x)
        return estimator
        
    def get_entries(self, flat_in):
        # calculates distances between inputs and codes
        matrix_similarity = tf.matmul(flat_in, self.embedding)
        d = (tf.reduce_sum(flat_in ** 2, axis=1, keepdims=True) 
             + tf.reduce_sum(self.embedding ** 2, axis=0)
            - 2 * matrix_similarity)
        
        # gets indices for the min values of the distance
        indices = tf.argmin(d, axis=1)
        return indices