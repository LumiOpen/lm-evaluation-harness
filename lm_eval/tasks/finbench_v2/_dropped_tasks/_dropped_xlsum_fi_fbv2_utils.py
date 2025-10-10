# Utility functions for the xlsum_fi_fbv2 summarization task.

import numpy as np
import traceback

# Try to import the necessary metrics from the 'evaluate' library.
try:
    import evaluate

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError(
        "Please install evaluation metrics via: pip install evaluate sacrebleu rouge_score bert_score transformers torch"
    )
except Exception as e:
    raise RuntimeError(
        f"Error loading evaluation metrics: {str(e)}. Please check your installation."
    )


def doc_to_text(doc) -> str:
    """
    Formats the document into the input prompt for the model.
    """
    return doc["text"]


def doc_to_target(doc) -> str:
    """
    Returns the ground truth summary from the document.
    """
    return doc["summary"]


def process_results_gen(doc, results):
    """
    Processes the model's generated results and calculates evaluation metrics.
    """
    pred = results[0].strip() if results and results[0] else ""
    refs_text = doc_to_target(doc)
    refs = [refs_text.strip()] if refs_text else [""]

    if not pred or not refs[0]:
        return {
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bert_score": 0.0,
        }

    try:
        bleu_results = bleu.compute(predictions=[pred], references=[refs])
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        bleu_results = {"bleu": np.NAN}

    try:
        rouge_results = rouge.compute(predictions=[pred], references=[refs])
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
        rouge_results = {"rouge1": np.NAN, "rouge2": np.NAN, "rougeL": np.NAN}

    try:
        # We must explicitly provide num_layers because the model path is not
        # in the bert-score library's internal mapping.
        # num_layers for TurkuNLP/bert-base-finnish-cased-v1 is 12.
        # For intfloat/multilingual-e5-large num_layers is 24.
        bert_scores = bertscore.compute(
            predictions=[pred],
            references=refs,
            model_type="TurkuNLP/bert-base-finnish-cased-v1",
            num_layers=12,
            batch_size=4,
            device=None,
        )["f1"]
    except Exception as e:
        print("\n--- BERTScore Calculation Error ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("Full Traceback:")
        traceback.print_exc()
        print(f"Prediction (len={len(pred)}): '{pred[:100]}...'")
        print(f"Reference (len={len(refs[0])}): '{refs[0][:100]}...'")
        print("--- End of Error Report ---\n")
        bert_scores = [np.NAN]

    if bleu_results.get("bleu") == 0.0:
        bleu_results["bleu"] = 1e-7

    return {
        "bleu": bleu_results["bleu"],
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bert_score": np.mean(bert_scores),
    }