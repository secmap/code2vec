import pyhash
import hashlib
import bitarray
import pickle

class bloomfilter():
    def __init__(self, size=65536, k=7, name='bf', load=False):
        if load:
            self.load(name)
        else:
            self.size = size
            if k > 18 or k <= 0:
                print('k should be > 0 & <= 18')
                return None
            self.k = k
            self.name = name
            self.bitarray = bitarray.bitarray('0'*self.size)
            self.tables = [ [ set() for j in range(self.size) ] for i in range(self.k) ]

        self.hashes = [ pyhash.fnv1_64(),
                        pyhash.murmur2_x64_64a(),
                        pyhash.murmur3_x64_128(),
                        pyhash.lookup3(),
                        pyhash.super_fast_hash(),
                        pyhash.city_128(),
                        pyhash.spooky_128(),
                        pyhash.farm_128(),
                        pyhash.metro_128(),
                        pyhash.mum_64(),
                        pyhash.t1_64(),
                        pyhash.xx_64(),
                        lambda str: int(hashlib.md5(str.encode('utf-8')).hexdigest(), 16),
                        lambda str: int(hashlib.sha1(str.encode('utf-8')).hexdigest(), 16),
                        lambda str: int(hashlib.sha224(str.encode('utf-8')).hexdigest(), 16),
                        lambda str: int(hashlib.sha256(str.encode('utf-8')).hexdigest(), 16),
                        lambda str: int(hashlib.sha384(str.encode('utf-8')).hexdigest(), 16),
                        lambda str: int(hashlib.sha512(str.encode('utf-8')).hexdigest(), 16)
                    ]
    def query(self, str):
        for i in range(self.k):
            if self.bitarray[self.hashes[i](str) % self.size] == False:
                return False
        return True

    def add(self, str):
        res = bitarray.bitarray('0'*self.size)
        index = []
        for i in range(self.k):
            index.append(self.hashes[i](str) % self.size)
            res[self.hashes[i](str) % self.size] = True
            self.bitarray[self.hashes[i](str) % self.size] = True
            self.tables[i][self.hashes[i](str) % self.size].add(str)

        return res, index

    def save(self):
        with open(self.name+'.plk', 'wb') as f:
            pickle.dump([self.k, self.size, self.bitarray, self.tables], f, pickle.HIGHEST_PROTOCOL)

    def load(self, file):
        with open(file+'.plk', 'rb') as f:
            bf = pickle.load(f)

        self.k = bf[0]
        self.size = bf[1]
        self.bitarray = bf[2]
        self.tables = bf[3]
        self.name = file
