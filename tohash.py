#!/usr/bin/python3
import subprocess
import sys
import magic
import os
import pickle
import hashlib
import re
from bf import bloomfilter
import argparse


ERROR = ['Read file error', 'Not a valid PE file']

def progress(count, total, suffix=''):
     bar_len = 60
     filled_len = int(round(bar_len * count / float(total)))

     percents = round(100.0 * count / float(total), 1)
     bar = '=' * filled_len + '-' * (bar_len - filled_len)

     sys.stdout.write('[%s] %s%s ...%s (%s/%s)\r' % (bar, percents, '%', suffix, count, total))
     sys.stdout.flush()  # As suggested by Rom Ruben


class Obj2hash():
    """
        class for generating hashes
        //maybe implemented by a function is ok.
    """
    def __init__(self, name, size, k):
        self.bloomfilter = bloomfilter(name=name, size=size, k=k)

    def obj2hash(self, file):
        output = subprocess.check_output(['objdump', '-M', 'intel', '-d', file])
        if output == None:
            print(ERROR[1], file=sys.stderr)
            return None

        output = output.decode('utf-8', 'ignore')
        output = output.split("\n")
        for i in range(len(output) - 1, -1, -1):
            if output[i] == '':
                del output[i]
            elif output[i].count("\t") < 2:
                del output[i]
            else:
                output[i] = output[i].split("\t")[1]
                output[i] = output[i].replace(' ', '')

        hash_list = []
        total = len(output)
        cnt = 0
        for i in output:
            vec, indice = self.bloomfilter.add(i)
            indice_tuple = ','.join([str(idx) for idx in indice])
            hash_list.append('{}'.format(indice_tuple))
            cnt+=1
            progress(cnt, total)
        return hash_list

    def save_table(self):
        self.bloomfilter.save()


def parse_arguments():
    """
        Handling the arguments
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("input", help="enter the file containting words", type=str)
    parser.add_argument("output", help="the output hash file name", type=str)
    parser.add_argument("-k",
             help="the total number of hash functions (def:7)",
             type=int,
             default=7)
    parser.add_argument("-max_bf_size","-bf",
             help="the max index of hash functions (def:2^16)",
             type=int,
             default=65535)
    return parser.parse_args()

def gen_hash(hasher, file_path, outf):
    """
        given a file containing words, return hash list
    """
    print('Processing: {}'.format(file_path))
    hash_list = hasher.obj2hash(file_path)
    if hash_list is not None:
        hash_list = [str(h) for h in hash_list]
        hash_str = '\n'.join(hash_list)
        outf.write(hash_str + '\n')
    else:
        print('Error for processing {}'.format(file_path))
    return


def is_valid_header(header):
    if 'PE' in header and 'executable' in header:
        return True
    return False


def main():
    args = parse_arguments()
    output_file = open(args.output + '_' + str(args.max_bf_size) + '.hash', 'w')

    tohash = Obj2hash(args.output, args.max_bf_size, args.k)
    try:
        if os.path.isdir(args.input):
            search_root = args.input
            for root, dirnames, filenames in os.walk(search_root):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    if is_valid_header(magic.Magic().id_filename(file_path)):
                        gen_hash(tohash, file_path, output_file)
        else:
            if is_valid_header(magic.Magic().id_filename(file_path)):
                gen_hash(tohash, args.input, output_file)
            else:
                print(ERROR[1])
                sys.exit(-1)
    finally:
        tohash.save_table()
        output_file.close()

if __name__ == "__main__":
    main()
