import datasets
import sacrebleu
import numpy as np

from rouge_score import rouge_scorer, scoring


def process_results_mc2(doc, results):
    lls, is_greedy = zip(*results)

    # Split on the first `0` as everything before it is true (`1`).
    split_idx = list(doc["mc2_targets"]["labels"]).index(0)
    # Compute the normalized probability mass for the correct answer.
    ll_true, ll_false = lls[:split_idx], lls[split_idx:]
    p_true, p_false = np.exp(np.array(ll_true)), np.exp(np.array(ll_false))
    p_true = p_true / (sum(p_true) + sum(p_false))

    return {"acc": sum(p_true)}