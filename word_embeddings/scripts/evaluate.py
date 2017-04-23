import sys

def load_labels(path):
  return [int(line.strip()) for line in open(path)]

def precision_and_recall(references, candidates):
  assert len(references) == len(candidates), "Must have same number of labels."

  false_positives, false_negatives = 0, 0
  true_positives, true_negatives = 0, 0
  correct_label = 0
  for i, reference in enumerate(references):
    if reference == -1:
      if candidates[i] == -1:
        true_negatives += 1
      else:
        false_positives += 1
    else:
      if candidates[i] != -1:
        true_positives += 1
        if candidates[i] == reference:
          correct_label += 1
      else:
        false_negatives += 1

  # Compute unlabelled results.
  precision = 0.0
  if true_positives + false_positives > 0.0:
    precision = float(true_positives) / (true_positives + false_positives)
  recall = 0.0
  if true_positives + false_negatives > 0.0:
    recall = float(true_positives) / (true_positives + false_negatives)
  fscore = 0.0
  if precision + recall > 0.0:
    fscore = 2 * (precision * recall) / (precision + recall)

  # And with labels.
  labelled_precision = 0.0
  if true_positives + false_positives > 0.0:
    labelled_precision = float(correct_label) / (true_positives + false_positives)
  labelled_recall = 0.0
  if true_positives + false_negatives > 0.0:
    labelled_recall = float(correct_label) / (true_positives + false_negatives)
  labelled_fscore = 0.0
  if labelled_precision + labelled_recall > 0.0:
    labelled_fscore = 2 * (labelled_precision * labelled_recall) / (
      labelled_precision + labelled_recall)

  return (labelled_precision, precision, labelled_recall, recall,
          labelled_fscore, fscore)

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print "Usage ./evaluate.py reference_labels candidate_labels"
    sys.exit(0)

  references, candidates = load_labels(sys.argv[1]), load_labels(sys.argv[2])
  print "Results: labeled (unlabelled) - precision %.3f (%.3f); recall %.3f (%.3f); fscore %.3f (%.3f)" % (
      precision_and_recall(references, candidates))
