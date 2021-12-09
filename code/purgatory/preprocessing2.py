import tensorflow as tf
import numpy as np
from functools import reduce


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """

    # TODO: load and concatenate training data from training file.
    with open(train_file) as f:
      train_words_list = f.read().split("\n")

    # TODO: load and concatenate testing data from testing file.
    with open(test_file) as f:
      test_words_list = f.read().split("\n")

    # FROM LAB:
    # Build Vocabulary (word id's)
    vocab = set(" ".join(train_words_list).split()) # collects all unique words in our dataset (vocab)
    word2id = {w: i for i, w in enumerate(list(vocab))} # maps each word in our vocab to a unique index (label encode)

    train_words = " ".join(train_words_list).split()
    test_words = " ".join(test_words_list).split()

    token_train = list(map(lambda s: word2id[s], train_words))
    token_test = list(map(lambda s: word2id[s], test_words))

    # BONUS: Ensure that all words appearing in test also appear in train

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.
    return (token_train, token_test, vocab)

a,b,c = get_data("data/train.txt","data/test.txt")
print(len(a))
print(len(b))
print(len(c))