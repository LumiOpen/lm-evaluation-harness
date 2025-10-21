"""Minimal HELMET utilities - just the essential process_results function."""


def helmet_process_results(doc, results):
    """Fix aggregation TypeError by computing exact_match and f1 properly."""
    # Use 'answer' field (we normalized all tasks to use 'answer')
    gold = str(doc["answer"]).strip() if doc.get("answer") else ""
    result = str(results[0]).strip() if results else ""

    exact_match = 1.0 if gold == result else 0.0

    # F1 score (token-level)
    pred_tokens = set(result.lower().split()) if result else set()
    gold_tokens = set(gold.lower().split()) if gold else set()

    if not gold_tokens and not pred_tokens:
        f1 = 1.0
    elif not gold_tokens or not pred_tokens:
        f1 = 0.0
    else:
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"exact_match": exact_match, "f1": f1}