#!/usr/bin/env python3

import sys
import bf
import collections
import tensorflow as tf
import re

if len(sys.argv) != 4:
    print('Usage:\n\tcount_frequency.py <input filename> <bloom filter plk> <output filename>')
    sys.exit(-1)


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
            most_common_words[item[0]] = item[1]

        for w in filter_set:
            if tuple(sorted(list(w))) in most_common_words:
                unsorted_res.append(list(w))
    
    del words
    del count
    del filter_set

    return unsorted_res, most_common_words

n_words = 50000

print('Read vocabulary from {}...'.format(sys.argv[1]), end='')
vocabulary, most_common_words = read_data(sys.argv[1], n_words)
print('Done')

print('Loading bloomfilter...', end='')
bloomfilter = bf.bloomfilter()
bloomfilter.load(sys.argv[2])
print('Done')

output_file = open(sys.argv[3], 'w')

for item in vocabulary:
    hash_indice = tuple(sorted(list(item)))
    count = most_common_words[hash_indice]

    possible_words = set()
    for idx, val in enumerate(hash_indice):
        if idx == 0:
            possible_words = bloomfilter.get_opcode_in_table(idx, val)
        else:
            possible_words &= bloomfilter.get_opcode_in_table(idx, val)
            
    if len(possible_words) == 0:
        print('Unable to find reversed words for: {}'.format(item))
    else:
        output_file.write('{}\t#{}\t{}\n'.format(possible_words, item, count))

output_file.close()
    