from collections import defaultdict

import numpy as np
from datasets import Dataset


def process_docs_fluency(dataset):
    """Present each annotated Finnish pair in both candidate orders."""
    rows = []
    for doc in dataset:
        for order, candidates, answer in (
            ("raw_first", (doc["raw_mt"], doc["target_pemt"]), "B"),
            ("target_first", (doc["target_pemt"], doc["raw_mt"]), "A"),
        ):
            rows.append(
                {
                    **doc,
                    "order": order,
                    "answer": answer,
                    "prompt": (
                        "Which Finnish text is more fluent and natural? "
                        "Answer with A or B only.\n\n"
                        f"Finnish text A: {candidates[0]}\n"
                        f"Finnish text B: {candidates[1]}"
                    ),
                }
            )
    return Dataset.from_list(rows)


def process_results_fluency(doc, results):
    """Score a bare A/B final answer, counting invalid answers as incorrect.

    Backends should strip reasoning with their native think_end_token setting
    before scoring. Explanations and unstripped reasoning are invalid answers.
    """
    answer = results[0].strip().upper()
    valid = answer in ("A", "B")
    acc = float(valid and answer == doc["answer"])
    return {
        "acc": acc,
        "pair_acc": (doc["pair_id"], acc),
        "invalid_answer_rate": float(not valid),
    }


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
