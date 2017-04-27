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


class TranslationModel:
    """Models conditional diposteriors.instribution over trg words
       given a src word."""

    @staticmethod
    def normalize(matrix):
        """ normalizes 'matrix' by rows
            type matrix: dict{str : dict {str : float}}
        """

        for src, matches in matrix.items():
            norm = sum(cnt for trg, cnt in matches.items())
            for trg, cnt in matches.items():
                matches[trg] = cnt / norm

        return matrix

    def __init__(self, src_corpus, trg_corpus):
        self._src_trg_counts = {}  # Statistics
        self._trg_given_src_probs = {}  # Parameters
        counts = count_word_cooccurrences(src_corpus, trg_corpus)
        self._trg_given_src_probs = TranslationModel.normalize(counts)

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
        self._trg_given_src_probs = TranslationModel.normalize(
            self._src_trg_counts
        )
        self._src_trg_counts = {}


class PriorModel:
    """Models the prior probability of an alignment given
       only the sentence lengths and token indices."""

    @staticmethod
    def normalize(matrix):
        for lengths_map, matches in matrix.items():
            norm = sum(cnt for indx_map, cnt in matches.items())
            for indx_map, cnt in matches.items():
                matches[indx_map] = float(cnt) / norm

        return matrix

    def map_lengths(self, src_len, trg_len):
        return (trg_len - src_len)

    def map_indexes(self, src_indx, trg_indx):
        return (trg_indx - src_indx)

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for more sophisticated models."
        counts = count_word_cooccurrences(src_corpus, trg_corpus)
        self._distance_probs = {}

        for src_sent, trg_sent in zip(src_corpus, trg_corpus):
            src_len, trg_len = len(src_sent), len(trg_sent)
            len_map = self.map_lengths(src_len, trg_len)
            self._distance_probs.setdefault(len_map, {})

            for src_indx, src_token in enumerate(src_sent):
                for trg_indx, trg_token in enumerate(trg_sent):
                    indx_map = self.map_indexes(src_indx, trg_indx)
                    self._distance_probs[len_map].setdefault(indx_map, 0)

                    self._distance_probs[len_map][indx_map] += counts[src_token][trg_token]

        self._distance_probs = PriorModel.normalize(self._distance_probs)
        self._new_distance_probs = {}

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a uniform priorself._distance_probs probability."
        len_map = self.map_lengths(src_length, trg_length)
        indx_map = self.map_indexes(src_index, trg_index)
        return self._distance_probs.get(len_map, 1).get(indx_map, 1)

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Extract the necessary statistics from this matrix if needed."
        len_map = self.map_lengths(src_length, trg_length)
        self._new_distance_probs.setdefault(len_map, {})
        for src_indx in range(src_length):
            for trg_indx in range(trg_length):
                indx_map = self.map_indexes(src_indx, trg_indx)
                self._new_distance_probs[len_map].setdefault(indx_map, 0)
                self._new_distance_probs[len_map][indx_map] += posterior_matrix[trg_indx][src_indx]

    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        self._distance_probs = PriorModel.normalize(self._new_distance_probs)
        self._new_distance_probs = {}
