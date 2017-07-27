#!/usr/bin/env python3

import tensorflow as tf
import sys
import os
import numpy as np
import bf
import pwn
import collections

if len(sys.argv) != 4:
    print('Usage:\n\tsimilarity.py <input filename> <bloom filter plk> <model filename>')
    sys.exit(-1)

print('Loading bloomfilter...', end='')
bloomfilter = bf.bloomfilter()
bloomfilter.load(sys.argv[2])
print('Done')

def get_asm_indice(asm_str):
    opcode = pwn.asm(asm_str, vma=0xdeadbeef)
    indice = bloomfilter.get_indice(opcode)
    return indice


def read_data(filename, n_words):
    """Extract the first file enclosed in a zip file as a list of words."""
    with open(filename) as f:
        filter_set = set()
        unsorted_res = []
        words = []
        count = []
        for line in f:
            word = line.strip()
            if len(word) == 0:
                continue
            word_idx_list = [int(idx) for idx in word.split(',')]
            filter_set.add(tuple(word_idx_list))
            words.append(tuple(sorted(word_idx_list)))
        words_counter = collections.Counter(words)
        most_common_words = dict()
        most_common_words_counter = words_counter.most_common(n_words)
        for item in most_common_words_counter:
            most_common_words[item[0]] = True

        for w in filter_set:
            if tuple(sorted(list(w))) in most_common_words:
                unsorted_res.append(list(w))
    
    del most_common_words
    del words
    del count
    del filter_set

    return unsorted_res

n_words = 50000

print('Read vocabulary from {}...'.format(sys.argv[1]), end='')
vocabulary = read_data(sys.argv[1], n_words)
print('Done')

embedding_size = 256 # Dimension of the embedding vector.

bloom_filter_max_size = 65536
num_hash_fun = 7

top_k = 8

print('Construct required tf graph...', end='')

graph = tf.Graph()

with graph.as_default():
    test_input = tf.placeholder(tf.int32, shape=[num_hash_fun])

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(
            tf.random_uniform([bloom_filter_max_size, embedding_size], -1.0, 1.0))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    test_vec = tf.nn.embedding_lookup(embeddings, test_input)
    test_vec = tf.reduce_mean(test_vec, 0)
    test_vec = tf.expand_dims(test_vec, 0)
    print('test_vec.shape: ', test_vec.shape)


    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    all_words_embeddings = tf.nn.embedding_lookup(normalized_embeddings, vocabulary) / tf.sqrt(tf.reduce_sum(tf.square(test_vec)))
    all_words_embeddings = tf.reduce_mean(all_words_embeddings, 1)

    similarity = tf.matmul(test_vec, all_words_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

print('Done')
print('Tensorflow session start')

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()

    print('Restore embeddings weights from model({})...'.format(sys.argv[3]), end='')
    saver = tf.train.Saver({'embeddings': embeddings})
    saver.restore(session, sys.argv[3])
    print('Done')
    
    while True:
        raw_user_input = input('Please input a word (idx1,idx2,...idx7) or asm or exit:')
        if raw_user_input == 'exit':
            break
        if raw_user_input.startswith('('):
            user_input = [int(i) for i in raw_user_input[1:-1].split(',')]
        else:
            user_input = get_asm_indice(raw_user_input)

        feed_dict = {test_input: user_input}
        sim = session.run([similarity], feed_dict=feed_dict)[0][0]
        nearest = (-sim).argsort()[1:top_k + 1]
        log_str = 'Nearest to <{}>:'.format(raw_user_input)
        for k in range(top_k): # Iterate each top_k closed word
            close_opcode_indice = vocabulary[nearest[k]] # all the hash value of the closed word
            opcode_str = ''
            opcodes = set()
            for idx, val in enumerate(close_opcode_indice):
                if idx == 0:
                    opcodes = bloomfilter.get_opcode_in_table(idx, val)
                else:
                    opcodes &= bloomfilter.get_opcode_in_table(idx, val)
            # opcode_asm = pwn.disasm(opcode)
            if len(opcodes) == 0:
                print('Unable to find reversed opcode for: {}'.format(close_opcode_indice))
            else:
                opcode_asm_list = []
                for opcode in opcodes:
                    opcode_asm = pwn.disasm(bytearray.fromhex(opcode))
                    opcode_asm_list.append(opcode_asm)
                print('\t' + '; '.join(opcode_asm_list))
        print('=' * 80)
