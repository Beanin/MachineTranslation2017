# Models for word alignment

def count_word_cooccurrences(src_corpus, trg_corpus):
    "Counts how many times each pair of source and target words occur together."
    counts = {}
    for i, src_sent in enumerate(src_corpus):
        for src in src_sent:
            if src not in counts:
                counts[src] = {}
            for trg in trg_corpus[i]:
                if trg not in counts[src]:
                    counts[src][trg] = 0
                counts[src][trg] += 1
    return counts


def normalize(matrix):
    """ normalizes 'matrix' by rows
        type matrix: dict{str : dict {str : float}}
    """

    for src, matches in matrix.items():
        norm = sum(cnt for trg, cnt in matches.items())
        for trg, cnt in matches.items():
            matches[trg] = cnt / norm

    return matrix


class TranslationModel:
    "Models conditional diposteriors.instribution over trg words given a src word."

    def __init__(self, src_corpus, trg_corpus):
        self._src_trg_counts = {}  # Statistics
        self._trg_given_src_probs = {}  # Parameters
        counts = count_word_cooccurrences(src_corpus, trg_corpus)
        self._trg_given_src_probs = normalize(counts)

    def get_conditional_prob(self, src_token, trg_token):
        "Return the conditional probability of trg_token given src_token."
        if src_token not in self._trg_given_src_probs:
            print('src not found', src_token)
            return 1.0
        if trg_token not in self._trg_given_src_probs[src_token]:
            print('trg not found', trg_token)
            return 1.0
        res = self._trg_given_src_probs[src_token][trg_token]
        assert res <= 1, "%f" % res
        return res

    def collect_statistics(self, src_tokens, trg_tokens, posterior_matrix):
        "Accumulate fractional alignment counts from posterior_matrix."
        assert len(posterior_matrix) == len(trg_tokens)
        for posterior, trg_token in zip(posterior_matrix, trg_tokens):
            assert len(posterior) == len(src_tokens)

            for src_token, proba in zip(src_tokens, posterior):
                if src_token not in self._src_trg_counts:
                    self._src_trg_counts[src_token] = {}

                if trg_token not in self._src_trg_counts[src_token]:
                    self._src_trg_counts[src_token][trg_token] = 0

                self._src_trg_counts[src_token][trg_token] += proba

    def recompute_parameters(self):
        "Reestimate parameters from counts then reset counters"
        self._trg_given_src_probs = normalize(self._src_trg_counts)
        self._src_trg_counts = {}



class PriorModel:
    """Models the prior probability of an alignment given
       only the sentence lengths and token indices."""

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for more sophisticated models."
        self._distance_counts = {}
        self._distance_probs = {}

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a uniform prior probability."
        return 1.0 / src_length

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Extract the necessary statistics from this matrix if needed."
        pass

    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        pass

