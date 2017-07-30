# code2vec

This is a project for code2Vec.


1. `python3 tohash.py [PATH to DATASET]`
    -v      showing debug msg
    -k      indicate how many hashes will be used
    -bf     indicate how many bits in bloom filter
    -h      the help menu
    EX: `python3 tohash.py text8`
    Will output a task name, e.g. [text8_65535_7]
    The task name is composed of dataset's basename, `bf`, and `k`.
    The program will create `./output`/[TASK NAME] folder for putting
        1. a hash list containing all words' hash
        2. a pickle file for bloom filter

2. `python3 word2vec_tensorflow.py [TASK NAME]`
    -v      showing debug msg
    -k      indicate how many hashes will be used
    -bf     indicate how many bits in bloom filter
    -bat    the batch size (for training)
    -epoch  the training epoch (for training)
    -lr     the learning rate (for training)
    -emb    word embedding size (for w2v)
    -neg    negative samples (for w2v)
    -sw     skip window (for skip-grams)
    -ns     skip words (for skip-grams)
    EX: `python3 word2vec_tensorflow.py text8_65535_7`
    Will output a model, saved by TensorFlow per 100,000 epoch.
    Or you can stop for saving the last model, named with suffix "_last".
    The model files contain:
        1. a checkpoint
        2. a *.index
        3. a *.meta
        4. a *.data
    The default output model will replace the origin.

3. `python3 similarity.py [MODEL NAME]`
    -v      showing debug msg
    -bf     the max index of hash functions (def:2^16)
    -noc    the number of most common (def:50000)
    -emb    the embedding size (def:128)
    -top    show top n nearnest words
    EX: `python3 similarity.py output/minitext8_65535_7/minitext8_65535_7_last_2508`
    So far, the program needs to indicate the embedding size.
    In the future, it should be automatically read in the given model.


