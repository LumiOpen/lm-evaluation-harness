"""Cultural diversity (CD), cultural robustness (CR) and Cultural Signal Purity (CSP).

All scores come from cosine distances between Qwen3-Embedding-8B embeddings of the
same question answered in different languages:

- CD: mean pairwise cross-language distance on unanchored prompts (no country).
- CR: 1 - mean pairwise cross-language distance on anchored prompts ("We live in X"),
  averaged per base question over its anchors.
- CSP (pair-level, the paper's headline metric): for each language pair, the Spearman
  correlation over questions between the unanchored distance U_q and the anchored
  distance A_q (averaged over anchors); CSP = 1 - mean(rho)^2.
- CSP question-level: the earlier formulation, one Spearman correlation between the
  per-question means of U and A over all language pairs; CSP = 1 - rho^2.
"""

import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances

from lm_eval.api.registry import register_aggregation


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        'Please install the required dependencies for this task with `pip install lm_eval["cultural_robustness"]` or `pip install sentence-transformers`'
    ) from None

eval_logger = logging.getLogger(__name__)

# Spearman correlations need at least this many questions to be meaningful.
_MIN_CSP_QUESTIONS = 5

# Global state: the scores need every response of both tasks together, so
# process_results only records responses and the aggregations compute from here.
_ALL_RESPONSES: list[dict[str, Any]] = []
_SCORES: dict[str, Any] | None = None
_EMBEDDING_MODEL = None


def reset_state() -> None:
    """Reset recorded responses and cached scores between evaluation runs."""
    global _ALL_RESPONSES, _SCORES
    _ALL_RESPONSES = []
    _SCORES = None


def record_response(entry: dict[str, Any]) -> None:
    """Store one model response for later aggregation."""
    global _SCORES
    response = entry.get("response")
    stored = dict(entry)
    stored["raw_response"] = response
    stored["response"] = "" if response is None else str(response).strip()
    _ALL_RESPONSES.append(stored)
    _SCORES = None


def _get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once (defaults to Qwen3-Embedding-8B)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        model_name = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = os.environ.get("EMBEDDING_DEVICE", default_device)
        if device in ("cuda", "cuda:all"):
            device = "cuda:0"
        eval_logger.info(f"Loading embedding model {model_name} on {device}")
        # Load on the CPU and move to the GPU in bf16. The evaluated model is still
        # on the GPU, and older transformers versions would otherwise put a
        # float32 copy (32 GB for Qwen3-Embedding-8B) there first.
        model = SentenceTransformer(model_name, device="cpu")
        if device != "cpu":
            model = model.to(device=device, dtype=torch.bfloat16)
        _EMBEDDING_MODEL = model
    return _EMBEDDING_MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed all texts in one batched pass."""
    model = _get_embedding_model()
    model_name = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
    encode_kwargs: dict[str, Any] = {
        "batch_size": int(os.environ.get("EMBEDDING_BATCH_SIZE", "64")),
        "show_progress_bar": True,
        "convert_to_numpy": True,
    }
    # Qwen3 embedding models expect the document prompt for passages.
    if "Qwen" in model_name:
        encode_kwargs["prompt_name"] = "document"
    eval_logger.info(f"Embedding {len(texts)} responses")
    return model.encode(texts, **encode_kwargs)


def _sort_key(value: str) -> tuple:
    try:
        return (0, int(value), "")
    except ValueError:
        return (1, 0, value)


def _group_responses(task_type: str) -> dict[str, dict[str, str]]:
    """Group non-empty responses by prompt id: {group_id: {language: response}}.

    Unanchored prompts are grouped by question id; anchored prompts by
    question-anchor id ("12-7"), so every anchor is compared across languages.
    """
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    skipped = 0
    for item in _ALL_RESPONSES:
        if item.get("type") != task_type:
            continue
        base_id, language, response = (
            item.get("base_id"),
            item.get("language"),
            item.get("response"),
        )
        if not response:
            skipped += 1
            continue
        if base_id is None or language is None:
            continue
        group_id = str(base_id)
        if task_type == "unspecific":
            group_id = group_id.split("-")[0]
        grouped[group_id].setdefault(language, response)
    if skipped:
        eval_logger.warning(f"{task_type}: skipped {skipped} empty responses")
    return grouped


def _score_task(task_type: str) -> dict[str, Any] | None:
    """Embed one task's responses and compute its cross-language distances.

    Returns per-question mean distances keyed by base question id (averaged over
    anchors for anchored prompts) and per-language-pair distances keyed by
    pair -> base question id (likewise averaged over anchors).
    """
    grouped = _group_responses(task_type)
    group_ids = [g for g in sorted(grouped, key=_sort_key) if len(grouped[g]) >= 2]
    if not group_ids:
        return None

    texts: list[str] = []
    for group_id in group_ids:
        texts.extend(grouped[group_id].values())
    embeddings = embed_texts(texts)

    question_dists: dict[str, list[float]] = defaultdict(list)
    pair_dists: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    start = 0
    for group_id in group_ids:
        languages = list(grouped[group_id])
        n = len(languages)
        dist = cosine_distances(embeddings[start : start + n])
        start += n
        upper = np.triu_indices(n, k=1)
        base_id = group_id.split("-")[0]
        question_dists[base_id].append(float(dist[upper].mean()))
        for i, j in zip(*upper, strict=True):
            pair = tuple(sorted((languages[i], languages[j])))
            pair_dists[pair][base_id].append(float(dist[i, j]))

    return {
        "question": {q: float(np.mean(d)) for q, d in question_dists.items()},
        "pairs": {
            pair: {q: float(np.mean(d)) for q, d in by_q.items()}
            for pair, by_q in pair_dists.items()
        },
    }


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < _MIN_CSP_QUESTIONS:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _compute_csp(unanchored: dict[str, Any], anchored: dict[str, Any]) -> dict:
    """Pair-level and question-level CSP from the two tasks' distances."""
    common = sorted(
        set(unanchored["question"]) & set(anchored["question"]), key=_sort_key
    )
    rho_question = _spearman(
        [unanchored["question"][q] for q in common],
        [anchored["question"][q] for q in common],
    )

    pair_rhos = {}
    for pair, u_by_q in unanchored["pairs"].items():
        a_by_q = anchored["pairs"].get(pair)
        if a_by_q is None:
            continue
        qs = sorted(set(u_by_q) & set(a_by_q), key=_sort_key)
        rho = _spearman([u_by_q[q] for q in qs], [a_by_q[q] for q in qs])
        if not np.isnan(rho):
            pair_rhos[pair] = rho
    mean_rho = float(np.mean(list(pair_rhos.values()))) if pair_rhos else float("nan")

    return {
        "csp": 1.0 - mean_rho**2,
        "mean_pair_rho": mean_rho,
        "n_pairs": len(pair_rhos),
        "csp_question_level": 1.0 - rho_question**2,
        "question_rho": rho_question,
        "n_questions": len(common),
    }


def _scores() -> dict[str, Any]:
    """Compute (once per run) every score from the recorded responses."""
    global _SCORES
    if _SCORES is not None:
        return _SCORES
    if not _ALL_RESPONSES:
        raise ValueError(
            "No responses recorded for cultural_robustness; process_results was never called."
        )

    tasks = {t: _score_task(t) for t in ("unspecific", "specific")}
    scores: dict[str, Any] = {}
    if tasks["unspecific"] is not None:
        scores["cd"] = float(np.mean(list(tasks["unspecific"]["question"].values())))
    if tasks["specific"] is not None:
        scores["cr"] = 1.0 - float(
            np.mean(list(tasks["specific"]["question"].values()))
        )
    if tasks["unspecific"] is not None and tasks["specific"] is not None:
        scores.update(_compute_csp(tasks["unspecific"], tasks["specific"]))

    eval_logger.info(f"cultural_robustness scores: {scores}")

    _SCORES = scores
    return scores


def _get(key: str) -> float:
    scores = _scores()
    if key not in scores:
        raise ValueError(
            f"cultural_robustness: no question has non-empty responses in at least 2 languages, "
            f"so {key} cannot be computed (note that --limit takes the first N rows, which are all one language)."
        )
    return scores[key]


def _get_csp(key: str) -> float:
    scores = _scores()
    if key not in scores:
        eval_logger.warning(
            f"{key} needs both cultural_robustness tasks in the same run "
            "(--tasks cultural_robustness); reporting NaN."
        )
        return float("nan")
    return scores[key]


@register_aggregation("cultural_diversity_agg")
def cultural_diversity_agg(items: list[float]) -> float:
    """CD: mean cross-language cosine distance on unanchored prompts."""
    return _get("cd")


@register_aggregation("cultural_robustness_agg")
def cultural_robustness_agg(items: list[float]) -> float:
    """CR: 1 - mean cross-language cosine distance on anchored prompts."""
    return _get("cr")


@register_aggregation("cultural_signal_purity_agg")
def cultural_signal_purity_agg(items: list[float]) -> float:
    """Pair-level CSP = 1 - (mean over language pairs of Spearman(U, A))^2."""
    return _get_csp("csp")


@register_aggregation("cultural_signal_purity_question_level_agg")
def cultural_signal_purity_question_level_agg(items: list[float]) -> float:
    """Question-level CSP = 1 - Spearman(mean U per question, mean A per question)^2."""
    return _get_csp("csp_question_level")
