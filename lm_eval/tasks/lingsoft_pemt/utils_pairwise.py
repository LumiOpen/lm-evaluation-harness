"""Pairwise variant of the 7-choice fluency blank task (FI_blank_7choice).

Each test item becomes 42 A/B prompts: every ordered pair of its seven
candidates, so each pair is asked in both orders. The five fewshot items become
one demonstration each (gold vs. one distractor, gold alternating A/B).

The model's preference in a record is diff = ll(" A") - ll(" B"). The
order-debiased preference for `a` over `b` is (diff(a, b) - diff(b, a)) / 2.
"""

import itertools
import random
from collections import defaultdict

import datasets


SEED = 42


def _texts(doc):
    return dict(zip(doc["candidate_ids"], doc["candidate_texts"], strict=True))


def _row(doc, first, second, gold_pos, texts):
    head = doc["prompt"].split("\n\nA. ", 1)[0]
    return {
        "item_id": doc["item_id"],
        "lang_code": doc["lang_code"],
        "first": first,
        "second": second,
        "gold_candidate": doc["gold_candidate"],
        "gold_pos": gold_pos,
        "target": max(gold_pos, 0),
        "prompt": f"{head}\n\nA. {texts[first]}\nB. {texts[second]}\nAnswer:",
        "choices": [" A", " B"],
    }


def process_docs(dataset):
    rows = []
    for doc in dataset:
        texts = _texts(doc)
        gold = doc["gold_candidate"]
        for a, b in itertools.permutations(sorted(texts), 2):
            gold_pos = 0 if a == gold else 1 if b == gold else -1
            rows.append(_row(doc, a, b, gold_pos, texts))
    return datasets.Dataset.from_list(rows)


def process_fewshot_docs(dataset):
    rng = random.Random(SEED)
    rows = []
    for k, doc in enumerate(dataset):
        texts = _texts(doc)
        gold = doc["gold_candidate"]
        distractor = rng.choice(sorted(c for c in texts if c != gold))
        a, b = (gold, distractor) if k % 2 == 0 else (distractor, gold)
        rows.append(_row(doc, a, b, k % 2, texts))
    return datasets.Dataset.from_list(rows)


def process_results(doc, results):
    (ll_a, _), (ll_b, _) = results
    record = (
        doc["item_id"],
        doc["first"],
        doc["second"],
        doc["gold_candidate"],
        ll_a - ll_b,
    )
    return {
        name: record
        for name in (
            "gold_acc",
            "gold_pair_acc",
            "gold_pair_acc_debiased",
            "item_beats_all",
            "item_copeland_acc",
            "first_choice_rate",
        )
    }


def _by_item(items):
    """item_id -> (gold, {(first, second): diff})."""
    out = {}
    for item_id, first, second, gold, diff in items:
        out.setdefault(item_id, (gold, {}))[1][(first, second)] = diff
    return out


def _debiased(diffs):
    """{(a, b) with a < b: preference for a} for pairs present in both orders."""
    out = {}
    for (a, b), d_ab in diffs.items():
        if a < b and (b, a) in diffs:
            out[(a, b)] = (d_ab - diffs[(b, a)]) / 2
    return out


def _wins(diffs):
    wins = defaultdict(float)
    for (a, b), pref_a in _debiased(diffs).items():
        wins[a] += 1.0 if pref_a > 0 else 0.5 if pref_a == 0 else 0.0
        wins[b] += 1.0 if pref_a < 0 else 0.5 if pref_a == 0 else 0.0
    return wins


def _nan_mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def agg_gold_acc(items):
    """Rate of picking gold over all records that contain the gold (chance 0.5)."""
    return _nan_mean(
        float((diff > 0) == (first == gold))
        for _, first, second, gold, diff in items
        if gold in (first, second)
    )


def agg_gold_pair_acc(items):
    """Gold-vs-distractor pairs where gold wins in BOTH orders (chance 0.25)."""
    scores = []
    for gold, diffs in _by_item(items).values():
        for (a, b), d_ab in diffs.items():
            if a == gold and (b, a) in diffs:
                scores.append(float(d_ab > 0 and diffs[(b, a)] < 0))
    return _nan_mean(scores)


def agg_gold_pair_acc_debiased(items):
    """Gold-vs-distractor pairs won after averaging both orders (chance 0.5)."""
    scores = []
    for gold, diffs in _by_item(items).values():
        for (a, b), pref_a in _debiased(diffs).items():
            if gold == a:
                scores.append(float(pref_a > 0))
            elif gold == b:
                scores.append(float(pref_a < 0))
    return _nan_mean(scores)


def agg_item_beats_all(items):
    """Items where gold wins (order-debiased) against every other candidate."""
    scores = []
    for gold, diffs in _by_item(items).values():
        pairs = [p for p in _debiased(diffs) if gold in p]
        if pairs:
            scores.append(float(_wins(diffs)[gold] == len(pairs)))
    return _nan_mean(scores)


def agg_item_copeland_acc(items):
    """Items where gold has the most order-debiased pairwise wins (ties split, chance 1/7)."""
    scores = []
    for gold, diffs in _by_item(items).values():
        wins = _wins(diffs)
        if not wins:
            continue
        best = max(wins.values())
        top = [c for c, w in wins.items() if w == best]
        scores.append(1.0 / len(top) if gold in top else 0.0)
    return _nan_mean(scores)


def agg_first_choice_rate(items):
    """Share of records answered " A" (0.5 = no position bias)."""
    return _nan_mean(float(diff > 0) for *_, diff in items)
