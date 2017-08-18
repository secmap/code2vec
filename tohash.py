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
import logging

ERROR = ['Read file error']
OUT_DIR = "./output/"
logging.basicConfig(level=logging.INFO,
        format=' [%(levelname)-8s] %(message)s')
log = logging.getLogger("tohash")

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

    def obj2hash(self, f):
        output = open(f, 'r').read()
        if output == None:
            log.error(ERROR)
            return None

        output = output.split()

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
    parser.add_argument("-verbose", "-v",
            help="show debug information",
            action='store_true')
    parser.add_argument("-k",
             help="the total number of hash functions (def:7)",
             type=int,
             default=7)
    parser.add_argument("-max_bf_size","-bf",
             help="the max index of hash functions (def:2^16)",
             type=int,
             default=65536)
    return parser.parse_args()

def gen_hash(hasher, file_path, outf):
    """
        given a file containing words, return hash list
    """
    log.info('Processing: {}'.format(file_path))
    hash_list = hasher.obj2hash(file_path)
    if hash_list is not None:
        hash_list = [str(h) for h in hash_list]
        hash_str = '\n'.join(hash_list)
        outf.write(hash_str + '\n')
    else:
        log.error('Error for processing {}'.format(file_path))
    return

def main():
    args = parse_arguments()
    log.debug(args)
    if(args.verbose):
        log.setLevel(logging.DEBUG)

    task_name = "{}_{}_{}".format(
            os.path.basename(args.input)
            ,args.max_bf_size,
            args.k)
    work_folder = OUT_DIR+task_name
    try:
        os.mkdir(work_folder)
    except:
        if(os.path.exists(work_folder) and os.path.isdir(work_folder)):
            log.warn("Dir exists, you will reuse this work folder")
        else:
            log.error("Cannot create work folder \'{}\'".format(work_folder))
            sys.exit(1)

    out_hash_fname = task_name+'.hash'
    output_file = open(work_folder+'/'+out_hash_fname, 'w')
    log.debug("Open "+work_folder+'/'+out_hash_fname +" for writing hashes.")
    log.critical("Task Name is :["+task_name+"], you need them to specify the task")
    tohash = Obj2hash(work_folder+'/'+ task_name, args.max_bf_size, args.k)
    try:
        if os.path.isdir(args.input):
            search_root = args.input
            for root, dirnames, filenames in os.walk(search_root):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    gen_hash(tohash, file_path, output_file)
        else:
            gen_hash(tohash, args.input, output_file)
    finally:
        tohash.save_table()
        output_file.close()

if __name__ == "__main__":
    main()
