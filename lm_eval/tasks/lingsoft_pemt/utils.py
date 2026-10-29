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


def process_results_rotation(doc, results):
    lls = [ll for ll, _ in results]
    acc = 1.0 if int(np.argmax(lls)) == int(doc["gold"]) else 0.0
    scores = list(zip(doc["candidate_ids"], lls, strict=True))
    return {
        "acc": acc,
        "rotation_acc": (doc["item_id"], doc["gold_candidate"], scores),
    }


def agg_rotation_acc(items):
    """Item-level accuracy after averaging each candidate over its positions.

    Every item appears once per rotation, with each candidate in a different
    letter slot. A candidate's score is the mean loglikelihood of its letter
    across rotations, so a pure letter-position bias cancels out; the item
    counts as correct when the gold candidate has the highest mean. Items are
    scored over whichever rotations are present (e.g. under --limit).
    """
    candidate_scores = defaultdict(lambda: defaultdict(list))
    gold = {}
    for item_id, gold_candidate, scores in items:
        gold[item_id] = gold_candidate
        for candidate_id, score in scores:
            candidate_scores[item_id][candidate_id].append(score)
    if not gold:
        return float("nan")
    correct = 0
    for item_id, by_candidate in candidate_scores.items():
        means = {c: sum(s) / len(s) for c, s in by_candidate.items()}
        correct += max(means, key=means.get) == gold[item_id]
    return correct / len(gold)
