import pytest
from datasets import Dataset

from lm_eval.tasks.lingsoft_pemt.utils import (
    agg_pair_acc,
    process_docs_fluency,
    process_results_fluency,
)


def test_fluency_orders_and_prompt_metadata():
    dataset = Dataset.from_list(
        [
            {
                "pair_id": "pair-1",
                "raw_mt": "Raakateksti.",
                "target_pemt": "Korjattu teksti.",
                "fluency_labels": ["SECRET_ANNOTATION"],
                "occurrences": [{"source_text": "SECRET_ENGLISH_SOURCE"}],
            }
        ]
    )
    docs = process_docs_fluency(dataset)
    assert len(docs) == 2
    assert docs[0]["order"] == "raw_first"
    assert docs[0]["answer"] == "B"
    assert docs[1]["order"] == "target_first"
    assert docs[1]["answer"] == "A"
    assert docs[0]["pair_id"] == docs[1]["pair_id"] == "pair-1"
    assert (
        "Finnish text A: Raakateksti.\nFinnish text B: Korjattu teksti."
        in docs[0]["prompt"]
    )
    assert (
        "Finnish text A: Korjattu teksti.\nFinnish text B: Raakateksti."
        in docs[1]["prompt"]
    )
    for doc in docs:
        assert "SECRET_ANNOTATION" not in doc["prompt"]
        assert "SECRET_ENGLISH_SOURCE" not in doc["prompt"]
        assert doc["fluency_labels"] == ["SECRET_ANNOTATION"]


@pytest.mark.parametrize(
    ("response", "acc", "invalid"),
    [
        ("B", 1.0, 0.0),
        (" b\n", 1.0, 0.0),
        ("A", 0.0, 0.0),
        ("", 0.0, 1.0),
        ("A or B", 0.0, 1.0),
        ("B because it is better", 0.0, 1.0),
        ("<think>A</think>B", 0.0, 1.0),
    ],
)
def test_fluency_answer_scoring(response, acc, invalid):
    scores = process_results_fluency({"pair_id": "p", "answer": "B"}, [response])
    assert scores == {
        "acc": acc,
        "pair_acc": ("p", acc),
        "invalid_answer_rate": invalid,
    }


def test_fluency_pair_accuracy_detects_position_bias_and_drops_incomplete_pairs():
    assert agg_pair_acc([("p", 0.0), ("p", 1.0)]) == 0.0
    assert agg_pair_acc([("p", 1.0), ("p", 1.0), ("incomplete", 0.0)]) == 1.0
