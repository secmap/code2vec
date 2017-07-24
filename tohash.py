#!/usr/bin/python3

import subprocess
import sys
import magic
import os
import pickle
import hashlib
import re
from bf import bloomfilter

USAGE = "Transfer binary file to per instruction hashes\nUsage: {} <binary or search-root-folder> <output-filename>"
ERROR = ["{} is not a PE file.", "objdump error.", "Obj2hash error."]

CONSTANT_SYM = r'{const}'


class Obj2hash():

    def __init__(self, name):
        self.bloomfilter = bloomfilter(name=name)

    def obj2hash(self, file):
        output = subprocess.check_output(
            ['objdump', '-M', 'intel', '-d', file])
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
        for i in output:
            vec, indice = self.bloomfilter.add(i)
            indice_tuple = ','.join([str(idx) for idx in indice])
            hash_list.append('{}'.format(indice_tuple))
        return hash_list

    def save_table(self):
        self.bloomfilter.save()


def is_valid_header(header):
    if 'PE' in header and 'executable' in header:
        return True
    return False


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
                    try:
                        if is_valid_header(magic.Magic().id_filename(file_path)) and file_size < 5242880:
                            hash_list = tohash.obj2hash(file_path)
                            if hash_list is not None:
                                hash_list = [str(h) for h in hash_list]
                                hash_str = ' '.join(hash_list)
                                output_file.write(hash_str + '\n')
                    except UnicodeEncodeError:
                        pass
                    except subprocess.CalledProcessError:
                        print('Unable to process {}'.format(file_path))
        finally:
            tohash.save_table()
            output_file.close()
    else:
        if not is_valid_header(magic.Magic().id_filename(file)):
            print(ERROR[0].format(sys.argv[1]), file=sys.stderr)
            sys.exit(1)

        tohash = Obj2hash(sys.argv[2])
        hash_list = tohash.obj2hash(file)
        if hash_list is not None:
            hash_list = [str(h) for h in hash_list]
            hash_str = ' '.join(hash_list)
            output_file.write(hash_str + '\n')
            tohash.save_table()
        else:
            print(ERROR[2], file=sys.stderr)
            sys.exit(1)
        output_file.close()


if __name__ == "__main__":
    main()
