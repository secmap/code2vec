#!/usr/bin/env python3

import sys
import bf
import collections
import tensorflow as tf
import pwn
import re

if len(sys.argv) != 4:
    print('Usage:\n\tcount_frequency.py <input filename> <bloom filter plk> <output filename>')
    sys.exit(-1)

def get_words_counter(words, num_of_most_common_word):
    """Process raw inputs into a dataset."""

    words_counter = collections.Counter(words).most_common(num_of_most_common_word)
    return words_counter


def get_asm_indice(asm_str):
    opcode = pwn.asm(asm_str)
    indice = bloomfilter.get_indice(opcode)
    return indice


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with open(filename) as f:
        res = []
        data = tf.compat.as_str(f.read()).split()
        for word in data:
            word_idx_list = [int(idx) for idx in word.split(',')]
            res.append(tuple(word_idx_list))
    return res


print('Read vocabulary from {}...'.format(sys.argv[1]), end='')
vocabulary = read_data(sys.argv[1])
print('Done')

print('Loading bloomfilter...', end='')
bloomfilter = bf.bloomfilter()
bloomfilter.load(sys.argv[2])
print('Done')

output_file = open(sys.argv[3], 'w')

counter = get_words_counter(vocabulary, 1000)

for item in counter:
    opcode_indice = list(item[0])
    count = item[1]
    opcodes = set()
    for idx, val in enumerate(opcode_indice):
        if idx == 0:
            opcodes = bloomfilter.get_opcode_in_table(idx, val)
        else:
            opcodes &= bloomfilter.get_opcode_in_table(idx, val)
    # opcode_asm = pwn.disasm(opcode)
    if len(opcodes) == 0:
        print('Unable to find reversed opcode for: {}'.format(opcode_indice))
    else:
        opcode_asm_list = []
        for opcode in opcodes:
            opcode_asm = pwn.disasm(bytearray.fromhex(opcode))
            opcode_asm_list.append(opcode_asm)
        output_file.write('\t#{}\t{}\n'.format(count, '; '.join(opcode_asm_list)))

output_file.close()
    