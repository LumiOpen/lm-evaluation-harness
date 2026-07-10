from collections import defaultdict

import numpy as np


def process_results_preference(doc, results):
    lls = [ll for ll, _ in results]
    gold = doc["choices"].index(doc["completion"])
    acc = 1.0 if int(np.argmax(lls)) == gold else 0.0
    return {"acc": acc, "pair_acc": (doc["pair_id"], acc)}


def agg_pair_acc(items):
    """Fraction of pairs answered correctly in BOTH candidate orders.

    Each test pair appears twice (raw_first / target_first); a model with a
    pure position bias scores 0.5 on record-level acc but 0.0 here. Pairs with
    only one order present (e.g. under --limit) are dropped.
    """
    pairs = defaultdict(list)
    for pair_id, correct in items:
        pairs[pair_id].append(correct)
    complete = [scores for scores in pairs.values() if len(scores) == 2]
    if not complete:
        return float("nan")
    return sum(1.0 for scores in complete if all(scores)) / len(complete)
