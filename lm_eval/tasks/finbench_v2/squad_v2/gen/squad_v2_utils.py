# Based on NorEval - https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/noreval

import datasets
import transformers.data.metrics.squad_metrics as squad_metrics


def process_results(doc, results):
    preds = results[0]
    reference = doc["answers"]["text"][0]
    f1_sum = squad_metrics.compute_f1(reference, preds)
    exact_match = squad_metrics.compute_exact(reference, preds)
    return {"f1": f1_sum, "exact_match": exact_match}