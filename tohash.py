#!/usr/bin/python3

import subprocess
import sys
import magic
import os
import pickle
import hashlib
import re
from bf import bloomfilter

USAGE = "Transfer text file to per instruction hashes\nUsage: {} <input text file or root folder> <output-filename>"
ERROR = ['Read file error']

class Obj2hash():

    def __init__(self, name):
        self.bloomfilter = bloomfilter(name=name)

    def obj2hash(self, file):
        output = open(file, 'r').read()
        if output == None:
            print(ERROR[1], file=sys.stderr)
            return None

        output = output.split()

        hash_list = []
        for i in output:
            vec, indice = self.bloomfilter.add(i)
            indice_tuple = ','.join([str(idx) for idx in indice])
            hash_list.append('{}'.format(indice_tuple))
        return hash_list

    def save_table(self):
        self.bloomfilter.save()


def main():
    if len(sys.argv) != 3:
        print(USAGE.format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    file = sys.argv[1]
    output_file = open(sys.argv[2] + '.txt', 'w')

    if os.path.isdir(file):
        search_root = file
        tohash = Obj2hash(sys.argv[2])
        try:
            for root, dirnames, filenames in os.walk(search_root):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    file_size = os.path.getsize(file_path)
                    print('Processing: {}'.format(file_path))
                    hash_list = tohash.obj2hash(file_path)
                    if hash_list is not None:
                        hash_list = [str(h) for h in hash_list]
                        hash_str = ' '.join(hash_list)
                        output_file.write(hash_str + '\n')
        finally:
            tohash.save_table()
            output_file.close()
    else:
        tohash = Obj2hash(sys.argv[2])
        hash_list = tohash.obj2hash(file)
        if hash_list is not None:
            hash_list = [str(h) for h in hash_list]
            hash_str = '\n'.join(hash_list)
            output_file.write(hash_str + '\n')
            tohash.save_table()
        else:
            sys.exit(1)
        output_file.close()


if __name__ == "__main__":
    main()
