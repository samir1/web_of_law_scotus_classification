from gensim.models.keyedvectors import KeyedVectors

# download GoogleNews-vectors-negative300.bin.gz
# from https://code.google.com/archive/p/word2vec/
# and uncompress to GoogleNews-vectors-negative300.bin
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.save_word2vec_format('GoogleNews-vectors-negative300.txt', binary=False)