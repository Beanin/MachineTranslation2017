import os, sys
from normalization import normalize

def all_ordered_pairs(tokens):
    # Returns all ordered pairs of tokens.
    pairs = []
    for i, first in enumerate(tokens[:-1]):
        for j, second in enumerate(tokens[i+1:]):
            pairs.append((first, second))
    return pairs

def normalized_kendalls_tau(reference, candidate):
    # Computes probability that pair of words are in same order as reference.
    all_candidate_pairs = all_ordered_pairs(candidate.split())
    candidate_pair_counts = dict(
        [(pair, all_candidate_pairs.count(pair)) for pair in all_candidate_pairs])
    found = 0
    for ref_pair in all_ordered_pairs(reference.split()):
        if ref_pair in candidate_pair_counts:
            if candidate_pair_counts[ref_pair] > 0:
                found += 1
                candidate_pair_counts[ref_pair] -= 1
    return float(found) / len(all_candidate_pairs)

def score_all_reorderings(references, candidates):
    # Compute average kendall's tau over all sentences.
    assert len(candidates) == len(references)
    scores = {}
    for i in range(len(candidates)):
        reference, candidate = normalize(references[i], candidates[i])
        assert reference and candidate, "Normalization failed!"
        scores[i] = normalized_kendalls_tau(reference, candidate)
    print("Average normalized kendalls tau was %1.3f on %d sentences." % (
        float(sum(scores.values())) / len(list(scores.keys())), len(list(scores.keys()))))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_reordering.py references candidates")
        sys.exit(0)
    references = [line.strip() for line in open(sys.argv[1])]
    candidates = [line.strip() for line in open(sys.argv[2])]
    score_all_reorderings(references, candidates)
