import itertools

import datasets
import pytest

from lm_eval.tasks.lingsoft_pemt import utils_pairwise as pw


CANDIDATES = list("ABCDEFG")
GOLD = "C"
METRICS = {
    "gold_acc": pw.agg_gold_acc,
    "gold_pair_acc": pw.agg_gold_pair_acc,
    "gold_pair_acc_debiased": pw.agg_gold_pair_acc_debiased,
    "item_beats_all": pw.agg_item_beats_all,
    "item_copeland_acc": pw.agg_item_copeland_acc,
    "first_choice_rate": pw.agg_first_choice_rate,
}


def _doc():
    order = CANDIDATES[3:] + CANDIDATES[:3]
    options = "\n".join(
        f"{letter}. text {c}" for letter, c in zip(CANDIDATES, order, strict=True)
    )
    return {
        "prompt": f"Instruction.\n\nPassage:\nContext.\n[BLANK]\n\n{options}\nAnswer:",
        "item_id": "item",
        "lang_code": "FI",
        "candidate_ids": order,
        "candidate_texts": [f"text {c}" for c in order],
        "gold_candidate": GOLD,
    }


def _scores(item_id, quality, a_bias):
    """Aggregate every metric for one item scored as ll(A) - ll(B) = bias + quality gap."""
    records = []
    for first, second in itertools.permutations(CANDIDATES, 2):
        doc = {
            "item_id": item_id,
            "first": first,
            "second": second,
            "gold_candidate": GOLD,
        }
        diff = a_bias + quality[first] - quality[second]
        records.append(
            pw.process_results(doc, [(diff, False), (0.0, False)])["gold_acc"]
        )
    return records


def _aggregate(records):
    return {name: agg(records) for name, agg in METRICS.items()}


def test_process_docs_expands_every_ordered_pair():
    rows = list(pw.process_docs(datasets.Dataset.from_list([_doc()])))
    assert len(rows) == 42
    assert {(r["first"], r["second"]) for r in rows} == set(
        itertools.permutations(CANDIDATES, 2)
    )
    for row in rows:
        assert row["prompt"] == (
            "Instruction.\n\nPassage:\nContext.\n[BLANK]\n\n"
            f"A. text {row['first']}\nB. text {row['second']}\nAnswer:"
        )
        expected = 0 if row["first"] == GOLD else 1 if row["second"] == GOLD else -1
        assert row["gold_pos"] == expected


def test_fewshot_alternates_gold_position():
    docs = datasets.Dataset.from_list([_doc() for _ in range(5)])
    rows = list(pw.process_fewshot_docs(docs))
    assert [r["target"] for r in rows] == [0, 1, 0, 1, 0]
    assert all(GOLD in (r["first"], r["second"]) for r in rows)


def test_letter_bias_cancels_after_order_averaging():
    # Always prefers " A" strongly; slightly prefers the gold over every distractor.
    quality = {c: 0.1 if c == GOLD else 0.0 for c in CANDIDATES}
    scores = _aggregate(_scores("item", quality, a_bias=5.0))
    assert scores["first_choice_rate"] == 1.0
    assert scores["gold_acc"] == 0.5
    assert scores["gold_pair_acc"] == 0.0
    assert scores["gold_pair_acc_debiased"] == 1.0
    assert scores["item_beats_all"] == 1.0
    assert scores["item_copeland_acc"] == 1.0


def test_indifferent_model_scores_copeland_chance():
    scores = _aggregate(_scores("item", {c: 0.0 for c in CANDIDATES}, a_bias=0.0))
    assert scores["item_copeland_acc"] == pytest.approx(1 / 7)
    assert scores["item_beats_all"] == 0.0


def test_copeland_is_item_level():
    best = {c: 1.0 if c == GOLD else 0.0 for c in CANDIDATES}
    worst = {c: -1.0 if c == GOLD else 0.0 for c in CANDIDATES}
    records = _scores("right", best, 0.0) + _scores("wrong", worst, 0.0)
    for record in records[42:]:
        assert record[0] == "wrong"
    # Distinct item ids keep the two items apart.
    assert pw.agg_item_copeland_acc(records) == 0.5
