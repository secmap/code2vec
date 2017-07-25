#!/usr/bin/env python3

import tensorflow as tf
import sys
import os
import numpy as np
import collections


if len(sys.argv) != 3:
    print('Usage:\n\tsimilarity.py <input filename> <model filename>')
    sys.exit(-1)

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  data = open(filename, 'r').read().split()
  return data

print('Read input file from {}...'.format(sys.argv[1]), end='')
vocabulary = read_data(sys.argv[1])
print('Done')

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

print('Build dataset...', end='')
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Done')

embedding_size = 128  # Dimension of the embedding vector.

top_k = 8

print('Construct required tf graph...', end='')

graph = tf.Graph()

with graph.as_default():
    test_input = tf.placeholder(tf.int32, shape=[1])

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(
                        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    test_vec = tf.nn.embedding_lookup(embeddings, test_input)
    print('test_vec.shape: ', test_vec.shape)


    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    similarity = tf.matmul(test_vec, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

print('Done')
print('Tensorflow session start')

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()

    print('Restore embeddings weights from model({})...'.format(sys.argv[2]), end='')
    saver = tf.train.Saver({'embeddings': embeddings})
    saver.restore(session, sys.argv[2])
    print('Done')
    
    while True:
        raw_user_input = input('Please input a word or exit:')
        if raw_user_input == 'exit':
            break
            
        user_input = [dictionary[raw_user_input]]

        feed_dict = {test_input: user_input}
        sim = session.run([similarity], feed_dict=feed_dict)[0][0]
        nearest = (-sim).argsort()[1:top_k + 1]
        log_str = 'Nearest to <{}>:'.format(raw_user_input)
        close_words = []
        for k in range(top_k): # Iterate each top_k closed word
            close_word = reverse_dictionary[nearest[k]] # all the hash value of the closed word
            close_words.append(close_word)
        print(', '.join(close_words))
        print('=' * 80)
