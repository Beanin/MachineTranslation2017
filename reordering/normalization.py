from string import punctuation
from itertools import permutations

UNIMPORTANT_WORDS = (
    ['a', 'an', 'the', ',', '.',',' '.', ';', '-', ')', '(', '/', '\'', '\"'])

def remove_words(sentence, words):
    return ' '.join([w for w in sentence.split() if w not in words])

def have_same_words(reference, candidate):
    reference_words = set(remove_words(reference, ['{{', '}}']).split())
    candidate_words = set(remove_words(candidate, ['{{', '}}']).split())
    return len(reference_words.difference(candidate_words)) == 0

def add_sentence_start_and_end(sentence):
    return '<s> %s </s>' % sentence

def get_all_spans(reference):
    spans = []
    inside_brackets = False
    span = []
    for token in reference.split():
        if token == '{{':
            inside_brackets = True
            continue
        if token == '}}':
            inside_brackets = False
            spans.append(span)
            span = []
            continue
        span.append(token)
        if not inside_brackets:
            spans.append(span)
            span = []
    if len(span):
        spans.append(span)
    return spans

def best_matching_permutation(candidate, span):
    for i, permutation in enumerate(permutations(span)):
        if ' '.join(permutation) in candidate:
            return ' '.join(permutation)
        if i == 100:
            break
    return ' '.join(span)

def normalize_for_brackets(reference, candidate):
    return ' '.join([
        best_matching_permutation(
                candidate, span) for span in get_all_spans(reference)])

def normalize(reference, candidate):
    assert have_same_words(reference, candidate)
    # Remove unimportant words and brackets from candidate.
    candidate = remove_words(candidate, UNIMPORTANT_WORDS + ['{{', '}}'])
    reference = remove_words(reference, UNIMPORTANT_WORDS)
    # Reorder spans in {{ }} in reference to match candidate if possible.
    reference = normalize_for_brackets(reference, candidate)
    # Add <s> and </s>.
    reference = add_sentence_start_and_end(reference)
    candidate = add_sentence_start_and_end(candidate)
    return reference, candidate
