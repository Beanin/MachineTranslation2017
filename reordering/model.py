import os
import numpy as np
import sys
# from sklearn.linear_model import LogisticRegression
# accuracy 0.771
# from sklearn.naive_bayes import MultinomialNB
# 0.776
# from sklearn.tree import DecisionTreeClassifier
# 0.77*
# from sklearn.ensemble import RandomForestClassifier
# 0.781
from xgboost import XGBClassifier
from sklearn.feature_extraction import DictVectorizer
from tree import Tree


def _getRanks(iterable):
    return [index for index, node in
            sorted(enumerate(iterable), key=lambda x:x[1].index)]


class Data(object):
    def __init__(self, childNum):
        self.vectorizer = DictVectorizer(sparse=False)
        self.size = childNum
        self.features = []
        self.labels = []

    def extractFeaturesFromNode(self, node, fit=True):
        if not node.children:
            raise RuntimeError

        categorial = dict(enumerate(
            [node.category] +
            [child.label for child in node.children] +
            [child.category for child in node.children]
        ))

        if (fit):
            self.vectorizer.fit([categorial])

        categorialVect = self.vectorizer.transform([categorial])[0]

        childIndexes = np.array([child.index for child in node.children])

        features = np.concatenate([
            categorialVect,
            childIndexes
        ])
        if (fit):
            self.features.append(features)

        return features

    def extractLabel(self, node, reordering, fit=True):
        reordered = [(reordering[child.index], i)
                     for i, child in enumerate(node.children)]
        res = tuple(i for _, i in sorted(reordered))
        if (fit):
            self.labels.append(res)

        return res


def _parseReordering(line):
    indexes = [int(chunk) for chunk in line.split() if chunk.isdigit()]
    return {ind - 1: num for num, ind in enumerate(indexes)}


def _extractSamples(tree, reorderingStr, samples):
    """ tree - Tree instance
        samples - dict : number of children -> list of samples
        returns samples
    """
    reordering = _parseReordering(reorderingStr)

    for node in tree.root.breadth_first():
        if not node.children:
            continue

        sz = len(node.children)
        data = samples.setdefault(sz, Data(sz))

        data.extractFeaturesFromNode(node)
        data.extractLabel(node, reordering)

    return samples


class ChildrenReorderingModel(object):
    def trainModel(self, data):
        df, raw_labels = np.array(data.features), data.labels

        # get numerical labels
        vocab = {}
        rev_vocab = []
        labels = []
        cnt = 0
        for lbl in raw_labels:
            if lbl in vocab:
                labels.append(vocab[lbl])
            else:
                vocab[lbl] = cnt
                rev_vocab.append(lbl)
                labels.append(cnt)
                cnt += 1

        self.estimator = XGBClassifier(
            n_estimators=100, max_depth=3,
            min_child_weight=1, subsample=0.7, colsample_bytree=0.7,
            colsample_bylevel=0.8
        )
        self.estimator.fit(df, labels)
        self.vocab = rev_vocab

    def __init__(self, data=None):
        if data:
            try:
                self.data = data
                self.trainModel(data)
                return
            except ValueError:
                print("Use initial order for %d" % data.size, file=sys.stderr)
                pass

        self.data = None

    def predict(self, head):
        if self.data is None:
            return head.children[:]

        features = self.data.extractFeaturesFromNode(head,
                                                     fit=False).reshape(1, -1)
        prediction = self.estimator.predict(features)
        return [head.children[i]
                for i in self.vocab[prediction[0]]]


def prepareData(prefix='train_model4', samples={}):
    parsesFile = os.path.join(prefix + '.parses')
    reordFile = os.path.join(prefix + '.reorderings_indices')

    for line, reordering in zip(open(parsesFile), open(reordFile)):
        t = Tree(line)
        _extractSamples(t, reordering, samples)

    return samples


def trainModels(samples):
    models = {}
    for childNum, data in samples.items():
        if childNum == 1 or len(data.features) < 10:
            continue

        print("Training model for %d children" % childNum, file=sys.stderr)
        models[childNum] = ChildrenReorderingModel(data)

    return models
