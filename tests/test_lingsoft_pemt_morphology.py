import math

from lm_eval.tasks.lingsoft_pemt.utils import agg_rotation_acc, process_results_rotation


CANDIDATES = ["target", "raw", "omorfi_0", "omorfi_1"]


def _rotations(item_id, letter_bias, candidate_pref):
    """Four rotation records scored by a model with a letter bias plus a candidate preference."""
    records = []
    for rotation in range(4):
        order = CANDIDATES[rotation:] + CANDIDATES[:rotation]
        doc = {
            "item_id": item_id,
            "gold": order.index("target"),
            "gold_candidate": "target",
            "candidate_ids": order,
        }
        results = [
            (letter_bias[slot] + candidate_pref[candidate], True)
            for slot, candidate in enumerate(order)
        ]
        records.append(process_results_rotation(doc, results))
    return records


def _aggregate(records):
    acc = sum(r["acc"] for r in records) / len(records)
    return acc, agg_rotation_acc([r["rotation_acc"] for r in records])


def test_letter_bias_cancels_across_rotations():
    # Always prefers "A" strongly; slightly prefers the professional form.
    records = _rotations(
        "item",
        [5.0, 0.0, 0.0, 0.0],
        {"target": 0.1, "raw": 0.0, "omorfi_0": 0.0, "omorfi_1": 0.0},
    )
    acc, rotation_acc = _aggregate(records)
    assert acc == 0.25
    assert rotation_acc == 1.0


def test_preferring_raw_mt_scores_zero():
    records = _rotations(
        "item", [0.0] * 4, {"target": 0.0, "raw": 1.0, "omorfi_0": 0.0, "omorfi_1": 0.0}
    )
    assert _aggregate(records) == (0.0, 0.0)


def test_rotation_acc_is_item_level():
    right = _rotations(
        "a", [0.0] * 4, {"target": 1.0, "raw": 0.0, "omorfi_0": 0.0, "omorfi_1": 0.0}
    )
    wrong = _rotations(
        "b", [0.0] * 4, {"target": 0.0, "raw": 0.0, "omorfi_0": 1.0, "omorfi_1": 0.0}
    )
    assert _aggregate(right + wrong)[1] == 0.5


def test_empty_is_nan():
    assert math.isnan(agg_rotation_acc([]))
