#!/usr/bin/python3

import subprocess
import sys
import magic
import os
import pickle
import hashlib
import re

USAGE = "Transfer binary file to per instruction hashes\nUsage: {} <binary or search-root-folder> <output-filename>"
ERROR = ["{} is not a PE file.", "objdump error."]

CONSTANT_SYM = r'{const}'

class Obj2hash():

    def __init__(self, h_size=13):
        self.h_size = 2**h_size
        self.hash_fn = lambda str: int(hashlib.md5(str.encode('utf-8')).hexdigest(), 16)% self.h_size
        self.h_table = [set() for i in range(self.h_size)]
        self.total = 0
        self.c_count = 0

    def obj2hash(self, file):
        output = subprocess.check_output(['objdump', '-M', 'intel', '-d', file])
        if output == None:
            print(ERROR[1], file=sys.stderr)
            return None

        output = output.decode('utf-8')
        output = output.split("\n")
        for i in range(len(output) - 1, -1, -1):
            if output[i] == '':
                del output[i]
            elif output[i].count("\t") < 2:
                del output[i]
            else:
                output[i] = output[i].split("\t")[2]
                tmp = output[i]
                output[i] = re.sub(r'\-(0x[\dabcdef]+)', r'+\1', output[i])
                output[i] = re.sub(r'\-(\d+)', r'+\1', output[i])
                output[i] = re.sub(r'(?!^)\b((0x)?)[0-9a-f]+\b', CONSTANT_SYM, output[i])
                output[i] = re.sub(r'(?!^)\b\d+\b', CONSTANT_SYM, output[i])
                output[i] = re.sub(r'<.*>$', r'', output[i])
                output[i] = re.sub(r'#.*$', r'', output[i])
                
        hash_list = []
        for i in output:
            h = self.hash_fn(i)
            if not i in self.h_table[h]:
                self.h_table[h].add(i)
                self.total += 1
                if len(self.h_table[h]) > 1:
                    self.c_count += 1
            hash_list.append(h)
        return hash_list

    def collision(self):
        print("{} {}".format(self.c_count, self.total))
        #print("{} occur in size {} hash table.".format(count, len(self.h_table)))

    def save_table(self, file):
        with open(file, 'wb') as output:
            pickle.dump(self.h_table, output, pickle.HIGHEST_PROTOCOL)

    def load_table(self, file):
        with open(file, 'rb') as output:
            self.h_table = pickle.load(output)


def is_valid_header(header):
    if 'PE' in header and 'executable' in header:
        return True
    return False


def main():
    if len(sys.argv) != 3:
        print(USAGE.format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    file = sys.argv[1]
    output_file = open(sys.argv[2], 'w')

    if os.path.isdir(file):
        search_root = file
        tohash = Obj2hash(22)
        try:
            for root, dirnames, filenames in os.walk(search_root):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    file_size = os.path.getsize(file_path)
                    if is_valid_header(magic.Magic().id_filename(file_path)) and file_size < 5242880:
                        hash_list = tohash.obj2hash(file_path)
                        if hash_list is not None:
                            hash_list = [str(h) for h in hash_list]
                            hash_str = ' '.join(hash_list)
                            output_file.write(hash_str + '\n')
        finally:
            tohash.save_table('hash.plk')
            tohash.collision()
            output_file.close()
    else:
        if not is_valid_header(magic.Magic().id_filename(file)):
            print(ERROR[0].format(sys.argv[1]), file=sys.stderr)
            sys.exit(1)

        tohash = Obj2hash(22)
        tohash.obj2hash(file)
        tohash.save_table('hash.plk')
        #tohash.load_table('hash.plk')
        tohash.collision()


if __name__ == "__main__":
    main()
