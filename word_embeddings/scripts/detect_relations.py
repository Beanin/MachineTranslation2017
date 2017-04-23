import sys
from embeddings import Vocab, WordEmbedding
import numpy as np

def load_example_sets(path):
    # Loads list of pairs per line.
    return [[tuple(pair.split()) for pair in line.strip().split('\t')] for line in open(path)]

def load_labels(path):
    # Loads a label for each line (-1 indicates the pairs do not form a relation).
    return [int(label) for label in open(path)]

def label_all_examples_as_not_a_relation(examples):
    return [-1 for example in examples]

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Usage ./detect_relations.py vocab_file embedding_file train_data train_labels test_data"
        sys.exit(0)

    # Load vocab and embedding (these are not used yet!)
    vocab = Vocab(sys.argv[1])
    embedding = WordEmbedding(vocab, sys.argv[2])

    # Loads training data and labels.
    training_examples = load_example_sets(sys.argv[3])
    training_labels = load_labels(sys.argv[4])
    assert len(training_examples) == len(training_labels), "Expected one label for each line in training data."

    # Load test examples and labels each set of pairs as 'not a relation' (-1)
    # This is not a good idea... You can definitely do better!
    test_examples = load_example_sets(sys.argv[5])
    for null_label in label_all_examples_as_not_a_relation(test_examples):
        print '%d' % null_label

    # TODO
    # 1. Try using the word embedding to discriminate between sets of pairs
    # that belong to a common relation and ones that don't (labelled -1).
    # E.g. training_example = [ (dog, dogs), (cat, cats), ... ], training_label = 3
    #      training_example = [ (dog, fishing), (cats, in), ...], training_label = -1
    #
    # 2. Once you have a model that can discriminate related pairs from unrelated ones,
    # try to further classify those that form a relation into a specific class. 
    # Note: all relations (labels) are observed at least one in the training data.

