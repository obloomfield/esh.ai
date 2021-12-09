import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, embedding_size

        self.vocab_size = vocab_size
        self.window_size = 20 # DO NOT CHANGE!
        self.embedding_size = 350 #TODO
        self.batch_size = 100 #TODO 
        self.rnn_size = 256

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN 
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, self.embedding_size)
        self.lstm = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.dense1 = tf.keras.layers.Dense(self.rnn_size, activation="leakyrelu ")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        
        #TODO: Fill in 
        layer1Output = self.embedding_layer(inputs)

        layer2Output, last_output, state = self.lstm(layer1Output, initial_state=initial_state)
        final_state = (last_output, state)

        layer3Output = self.dense1(layer2Output)

        layer4Output = self.dense2(layer3Output)

        return layer4Output, final_state

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,probs,from_logits=False))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    #TODO: Fill in
    last_mult_window = (len(train_inputs) // model.window_size ) * model.window_size
    train_inputs = tf.reshape([train_inputs[:last_mult_window]], [-1, model.window_size]) # doesn't go past the last multiple of window size
    train_labels = tf.reshape([train_labels[:last_mult_window]], [-1, model.window_size])

    for i in range(0, len(train_labels), model.batch_size):
      cur_inputs = train_inputs[i:i + model.batch_size]
      cur_labels = train_labels[i:i + model.batch_size]

      with tf.GradientTape() as tape:
        pred, _ = model.call(cur_inputs, None)
        loss = model.loss(pred, cur_labels)
        print("loss: " + str(loss))

        # if i//model.batch_size % 4 == 0:
        #   train_acc = model.accuracy(pred, cur_labels)
        #   print("Accuracy on training set after {} training steps: {}".format(i, train_acc))
      
      gradients = tape.gradient(loss, model.trainable_variables)
      model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    pass


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    
    #TODO: Fill in
    #NOTE: Ensure a correct perplexity formula (different from raw loss)
    last_mult_window = (len(test_inputs) // model.window_size ) * model.window_size
    test_inputs = tf.reshape([test_inputs[:last_mult_window]], [-1, model.window_size]) # doesn't go past the last multiple of window size
    test_labels = tf.reshape([test_labels[:last_mult_window]], [-1, model.window_size])

    cnt = 0
    tot_loss = 0
    for i in range(0, len(test_labels), model.batch_size):
      cur_inputs = test_inputs[i:i + model.batch_size]
      cur_labels = test_labels[i:i + model.batch_size]
      pred, _ = model.call(cur_inputs, None)
      tot_loss += model.loss(pred, cur_labels)
      cnt += 1
    
    tot_loss /= cnt
    return tf.exp(tot_loss)
    pass  


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    #NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see impossibly small perplexities.
    
    train_words, test_words, vocab = get_data("data/test.txt","data/train.txt")
    # TODO:  Separate your train and test data into inputs and labels
    m = Model(len(vocab))
    train_x, train_y, test_x, test_y = [], [], [], []

    for i in range(0, len(train_words) -1, m.window_size):
        train_x.append(train_words[i:i+m.window_size])
        train_y.append(train_words[i+1:i+m.window_size+1])
    for i in range(0, len(test_words) -1, m.window_size):
        test_x.append(test_words[i:i+m.window_size])
        test_y.append(test_words[i+1:i+m.window_size+1])


    # TODO: initialize model
    
    # TODO: Set-up the training step
    # for j in range(m.num_epochs):
    #   print('Epoch: ', j)
    #   train(m,train_x,train_y)
    # TODO: Set up the testing steps
    train(m,train_x,train_y)
    perp = test(m,test_x,test_y)
    # Print out perplexity 
    print(perp)

    # BONUS: Try printing out various sentences with different start words and sample_n parameters 
    
    pass

if __name__ == '__main__':
    main()
