import sys
from embeddings import Vocab, WordEmbedding
from numpy import linalg as la
import numpy as np
from sklearn.svm import LinearSVC


def cosine(x, y):
    # Cosine of angle between vectors x and y
    return x.dot(y) / (la.norm(x) * la.norm(y))


def load_example_sets(path):
    # Loads list of pairs per line.
    return [[tuple(pair.split())
             for pair in line.strip().split('\t')]
            for line in open(path)]


def load_labels(path):
    # Loads a label for each line
    # (-1 indicates the pairs do not form a relation).
    return [int(label) for label in open(path)]


def label_all_examples_as_not_a_relation(examples, embedding):
    return [-1 for example in examples]


def extractFeatures(sample, embedding):
    embeddingDiffs = np.array(
        [embedding.Projection(word1) - embedding.Projection(word2)
         for word1, word2 in sample]
    )
    meanDiff = np.mean(embeddingDiffs, axis=0)
    avgCosineDiff = max([cosine(diff, meanDiff)
                         for diff in embeddingDiffs]) / len(embeddingDiffs)

    return np.concatenate([
        meanDiff,
        np.std(embeddingDiffs, axis=0),
        [avgCosineDiff]]
    )


def classifyByLogisticRegression(trainData, trainLabels, examples, embedding):
    estimator = LinearSVC()
    trainFeatures = np.array([extractFeatures(sample, embedding)
                              for sample in trainData])
    estimator.fit(trainFeatures, trainLabels)
    testFeatures = np.array([extractFeatures(sample, embedding)
                             for sample in examples])
    return estimator.predict(testFeatures)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage ./detect_relations.py vocab_file embedding_file"
              " train_data train_labels test_data")
        sys.exit(0)

    # Load vocab and embedding (these are not used yet!)
    vocab = Vocab(sys.argv[1])
    embedding = WordEmbedding(vocab, sys.argv[2])

    # Loads training data and labels.
    training_examples = load_example_sets(sys.argv[3])
    training_labels = load_labels(sys.argv[4])
    assert len(training_examples) == len(
        training_labels), "Expected one label for each line in training data."

    # Load test examples and labels each set of pairs as 'not a relation' (-1)
    # This is not a good idea... You can definitely do better!
    test_examples = load_example_sets(sys.argv[5])
    # for null_label in label_all_examples_as_not_a_relation(test_examples):
    #    print('%d' % null_label)

    for label in classifyByLogisticRegression(
        training_examples, training_labels, test_examples, embedding
    ):
        print('%d' % label)
