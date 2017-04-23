from math import sqrt
import sys
import numpy as np
from numpy import linalg as la

class Vocab:
    def __init__(self, vocab_file):
        word_counts = [(line.strip().split()[0].decode('utf8'), int(line.strip().split()[1]))
                       for line in open(vocab_file)]
        self.words = [word for word, count in word_counts]
        self.counts = [count for word, count in word_counts]
        self.indices = dict([(word, i) for i, word in enumerate(self.words)])

    def InVocab(self, word):
        return word in self.indices

    def Size(self):
        return len(self.words)

class WordEmbedding:
    def __init__(self, vocab, embedding_file, normalize=True):
        self.vocab = vocab
        self.matrix = np.loadtxt(embedding_file, dtype='float')
        if normalize:
            self.matrix /= la.norm(self.matrix, axis=1)[:,None]
    
    def Row(self, i):
        return self.matrix[i]

    def Projection(self, word):
        return self.matrix[self.vocab.indices[word]]

    def NumWords(self):
        return self.matrix.shape[0]

    def Size(self):
        return self.matrix.shape[1]

class Model:
    def __init__(self, vocab, embedding, n=20):
        # Matrix of embedding vectors for vocab
        self.embedding = embedding
        self.vocab = vocab
        # Number of candidates to return
        self.n = n
        # Expect one embedding vector for each word
        assert self.embedding.NumWords() == self.vocab.Size()

    def cosine(self, x, y):
        # Cosine of angle between vectors x and y
        return x.dot(y) / la.norm(x) * la.norm(y)

    def sort_words_by_cosine(self, candidate_vector):
        # Rank words in vocab by distance to candidiate_vector
        return sorted([(self.cosine(self.embedding.Projection(word), candidate_vector), word)
                          for word in self.vocab.words], reverse=True)

    def find_closest_n(self, word):
        # Find closest n words to word
        return self.sort_words_by_cosine(self.embedding.Projection(word))[:self.n]

    def find_analogy(self, word1, word1_prime, word2):
        # Find n best candidates for word2_prime (using a function of your choice)
        assert False, "Use sort_words_by_cosine to get candidates for word2_prime"
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: ./embeddings.py vocab_file embedding_file"
        sys.exit(0)
    vocab_file, embedding_file = sys.argv[1:3]
    vocab = Vocab(vocab_file)
    print "Loaded vocabulary"
    embedding = WordEmbedding(vocab, embedding_file, normalize=True)
    print "Loaded embeddings"
    model = Model(vocab, embedding)

    while True:
        words = raw_input("Please enter one or three space separated words: ").split()
        for word in words:
            if not vocab.InVocab(word):
                print "%s is not in this model."
                words = []
        if len(words) == 3:
            word1, word1_prime, word2 = words
            for distance, word2_prime in model.find_analogy(word1, word1_prime, word2):
                print '%s\t%.3f' % (word2_prime, distance)
        if len(words) == 1:
            for distance, neighbour in model.find_closest_n(word):
                print '%s\t%.3f' % (neighbour, distance)

