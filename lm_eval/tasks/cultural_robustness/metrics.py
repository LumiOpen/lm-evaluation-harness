"""Cultural robustness evaluation metric."""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        'Please install the required dependencies for this task with `pip install lm_eval["cultural_robustness"]` or `pip install sentence-transformers`'
    )

from lm_eval.api.registry import register_aggregation


# global state needed because clustering requires all responses together
_ALL_RESPONSES: List[Dict[str, Any]] = []
_EMBEDDING_MODEL = None


def reset_state() -> None:
    """Reset cached responses between evaluation runs."""
    global _ALL_RESPONSES
    _ALL_RESPONSES = []


reset_state()


def record_response(entry: Dict[str, Any]) -> None:
    """Store one model response for later aggregation."""
    response = entry.get("response")
    if response is None:
        cleaned = ""
    elif isinstance(response, str):
        cleaned = response.strip()
    else:
        cleaned = str(response).strip()

    stored = dict(entry)
    stored["raw_response"] = response
    stored["response"] = cleaned
    _ALL_RESPONSES.append(stored)


def _get_embedding_model():
    """Load embedding model (defaults to Qwen3-Embedding-8B)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        model_name = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        requested_device = os.environ.get("EMBEDDING_DEVICE", default_device)

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        use_data_parallel = gpu_count > 1

        # Normalize cuda:all or cuda to cuda:0
        if requested_device in ("cuda", "cuda:all"):
            requested_device = "cuda:0"

        print(f"Loading embedding model {model_name} on {requested_device}")
        if use_data_parallel:
            print(
                f"  Multi-GPU: SentenceTransformer will use {gpu_count} GPUs automatically"
            )

        model = SentenceTransformer(model_name, device=requested_device)

        _EMBEDDING_MODEL = {
            "model": model,
            "device": requested_device,
        }
        print("Embedding model ready.")

    return _EMBEDDING_MODEL


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts using SentenceTransformer (matches Maria's implementation)."""
    model_bundle = _get_embedding_model()
    model = model_bundle["model"]

    batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))

    # Build encode kwargs - only pass prompt_name for Qwen models
    model_name = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
    encode_kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }

    # Qwen3 embedding models require the prompt_name parameter
    if "Qwen" in model_name:
        encode_kwargs["prompt_name"] = "document"

    embeddings = model.encode(texts, **encode_kwargs)

    return embeddings


def find_optimal_k(embeddings: np.ndarray, min_k: int = 2, max_k: int = None) -> tuple:
    """Find optimal number of clusters (K range 2 to n-1 per paper)"""
    n_items = len(embeddings)

    if n_items < 3:
        return 2, 0.0, [0] * n_items

    # Set max_k to n-1 (not n) per paper
    if max_k is None:
        max_k = n_items - 1
    else:
        max_k = min(max_k, n_items - 1)

    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    best_k, best_score = 2, 0.0
    cluster_labels = []

    for k in range(min_k, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_scaled)

            if len(set(labels)) <= 1:
                score = 0.0
            else:
                score = silhouette_score(embeddings_scaled, labels)

        except Exception:
            score = 0.0
            labels = [0] * n_items

        if score > best_score:
            best_score, best_k = score, k
            cluster_labels = labels

    return best_k, best_score, cluster_labels


def _group_responses_by_question(task_type=None):
    grouped = defaultdict(dict)
    skipped = 0

    for item in _ALL_RESPONSES:
        # Filter by task type if specified
        if task_type and item.get("type") != task_type:
            continue

        base_id = item.get("base_id")
        language = item.get("language")
        response = (item.get("response") or "").strip()

        if not response:
            skipped += 1
            continue
        if base_id is None or language is None:
            continue

        # robustness keeps variations separate (0-1, 0-2), diversity groups them (just 0)
        if task_type == "specific" and "-" in str(base_id):
            group_key = str(base_id)  # keep full id like "0-1", "0-2"
        else:
            # diversity strips variation if present
            if "-" in str(base_id):
                group_key = str(base_id).split("-")[0]
            else:
                group_key = str(base_id)

        grouped[group_key].setdefault(language, response)

    return grouped, skipped


def _build_id_fields(base_id: str, task_type: str) -> Dict[str, str]:
    """Build ID-related fields based on task type."""
    if task_type == "specific" and "-" in str(base_id):
        base_part, variation_part = str(base_id).split("-", 1)
        return {
            "sentence_id": base_part,  # Always include sentence_id for compatibility
            "full_id": str(base_id),
            "base_id": base_part,
            "variation_id": variation_part,
        }
    else:
        return {"sentence_id": str(base_id)}


def _save_cluster_outputs(
    results: List[Dict[str, Any]],
    cluster_assignments: List[Dict[str, Any]],
    language_clusters: List[Dict[str, Any]],
    task_type: str,
    avg_score: float,
) -> None:
    """Save cluster output CSVs and summary JSON to OUTPUT_DIR/clustering/{task_type}/"""
    output_dir_env = os.environ.get("OUTPUT_DIR")
    if not output_dir_env:
        print("⚠️  OUTPUT_DIR not set, cluster outputs not saved")
        return

    if pd is None:
        print("⚠️  pandas not available, cluster outputs not saved")
        return

    try:
        output_dir = Path(output_dir_env) / "clustering" / task_type
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis_type = "robustness" if task_type == "specific" else "diversity"

        # Save results CSV
        if results:
            results_df = pd.DataFrame(results)
            results_path = output_dir / f"{analysis_type}_results.csv"
            results_df.to_csv(results_path, index=False)
            print(f"💾 Saved {len(results)} results to {results_path}")

        # Save cluster assignments CSV
        if cluster_assignments:
            assignments_df = pd.DataFrame(cluster_assignments)
            assignments_path = output_dir / f"{analysis_type}_cluster_assignments.csv"
            assignments_df.to_csv(assignments_path, index=False)
            print(
                f"💾 Saved {len(cluster_assignments)} cluster assignments to {assignments_path}"
            )

        # Save language clusters CSV
        if language_clusters:
            clusters_df = pd.DataFrame(language_clusters)
            clusters_path = output_dir / f"{analysis_type}_language_clusters.csv"
            clusters_df.to_csv(clusters_path, index=False)
            print(
                f"💾 Saved {len(language_clusters)} language clusters to {clusters_path}"
            )

        # Save summary JSON (match Maria's format)
        model_name = os.environ.get(
            "MODEL_NAME", os.environ.get("MODEL_ID", "unknown_model")
        )

        if task_type == "specific":
            summary = {
                "model": model_name,
                "analysis_type": "robustness",
                "total_sentence_variation_combinations": len(results),
                "average_relative_score": float(avg_score),
                "duplicates_detected": 0,  # always 0, kept for format compatibility
                "total_results": len(results),
            }
        else:
            summary = {
                "model": model_name,
                "analysis_type": "diversity",
                "total_sentence_ids": len(results),
                "average_relative_score": float(avg_score),
                "duplicates_detected": 0,  # always 0, kept for format compatibility
                "total_results": len(results),
            }

        summary_path = output_dir / f"{analysis_type}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Saved summary to {summary_path}")

    except Exception as e:
        print(f"⚠️  Error saving cluster outputs: {e}")


def _compute_metric(task_type: str) -> float:
    """Compute clustering metric for a specific task type.

    Args:
        task_type: Either "unspecific" (diversity) or "specific" (robustness)

    Raises:
        ValueError: If no responses or invalid data
    """
    if not _ALL_RESPONSES:
        raise ValueError(
            "No responses collected for cultural robustness evaluation. "
            "This indicates a bug in the evaluation pipeline."
        )

    if task_type not in ["unspecific", "specific"]:
        raise ValueError(f"Invalid task_type: {task_type}")

    grouped, skipped = _group_responses_by_question(task_type)

    if not grouped:
        raise ValueError(
            f"No usable responses after grouping for task_type={task_type}. "
            f"Total responses: {len(_ALL_RESPONSES)}, Skipped (empty): {skipped}"
        )

    # Filter languages by task type (explicit, not inferred)
    languages = sorted(
        {
            item.get("language")
            for item in _ALL_RESPONSES
            if item.get("language") and item.get("type") == task_type
        }
    )

    # Analysis type based on explicit task_type (no auto-detection)
    analysis_type = "Robustness" if task_type == "specific" else "Diversity"

    print(f"\n{'=' * 60}")
    print(f"{analysis_type.upper()} ANALYSIS (DETAILED)")
    print(f"{'=' * 60}")

    k_values: List[float] = []
    relative_scores: List[float] = []

    # Track data for CSV outputs (like Maria's script)
    results: List[Dict[str, Any]] = []
    cluster_assignments: List[Dict[str, Any]] = []
    language_clusters: List[Dict[str, Any]] = []

    def _sort_key(value: Any) -> tuple:
        # Return tuple (is_numeric, numeric_value, string_value) for consistent sorting
        try:
            return (0, int(value), "")
        except (TypeError, ValueError):
            return (1, 0, str(value))

    sorted_ids = sorted(grouped.keys(), key=_sort_key)
    id_iterator = tqdm(
        sorted_ids, desc=f"Clustering {analysis_type.lower()}", unit="question"
    )

    for base_id in id_iterator:
        lang_map = grouped[base_id]
        n_items = len(lang_map)
        if n_items < 2:
            continue

        languages = list(lang_map.keys())
        sentences = list(lang_map.values())
        try:
            embeddings = embed_texts(sentences)
            optimal_k, silhouette, cluster_labels = find_optimal_k(embeddings)

            # calculate normalized score: (k-2)/(n-3) per paper, where k ∈ [2, n-1]
            if n_items < 3:
                clustering_score = 0.0
            else:
                clustering_score = (optimal_k - 2) / (n_items - 3)

            # diversity wants more clusters (higher), robustness wants fewer (invert)
            if task_type == "specific":
                relative_score = 1 - clustering_score
            else:
                relative_score = clustering_score

            k_values.append(optimal_k)
            relative_scores.append(relative_score)

            # Track which languages cluster together
            cluster_groups = defaultdict(list)
            for language, cluster_id in zip(languages, cluster_labels):
                cluster_groups[cluster_id].append(language)

            # Build common metrics dict
            common_metrics = {
                "n_languages": n_items,
                "total_items": n_items,  # No duplicates in lm-eval
                "languages": ", ".join(sorted(languages)),
                "optimal_k": int(optimal_k),
                "silhouette_score": float(silhouette),
                "relative_score": float(relative_score),
            }

            # Store results
            result_entry = {**_build_id_fields(base_id, task_type), **common_metrics}
            results.append(result_entry)

            # Store cluster assignments
            for language, cluster_id in zip(languages, cluster_labels):
                assignment_entry = {
                    **_build_id_fields(base_id, task_type),
                    "language": language,
                    "cluster_id": int(cluster_id),
                    "optimal_k": int(optimal_k),
                    "relative_score": float(relative_score),
                }
                cluster_assignments.append(assignment_entry)

            # Store language clusters (multi-language clusters only)
            for cluster_id, cluster_langs in cluster_groups.items():
                if len(cluster_langs) > 1:
                    cluster_entry = {
                        **_build_id_fields(base_id, task_type),
                        "cluster_id": int(cluster_id),
                        "languages_in_cluster": ", ".join(sorted(cluster_langs)),
                        "cluster_size": len(cluster_langs),
                        "optimal_k": int(optimal_k),
                        "relative_score": float(relative_score),
                    }
                    language_clusters.append(cluster_entry)

            score_desc = (
                f"1-({optimal_k}-2)/({n_items}-3)"
                if task_type == "specific"
                else f"({optimal_k}-2)/({n_items}-3)"
            )
            print(
                f"📝 ID {base_id}: {n_items} languages → {optimal_k} clusters → score: {relative_score:.3f} [{score_desc}]"
            )
            print(f"   Languages: {', '.join(sorted(languages))}")

            # Show which languages clustered together (only multi-language clusters)
            for cluster_id, cluster_languages in sorted(cluster_groups.items()):
                if len(cluster_languages) > 1:
                    print(
                        f"   Cluster {cluster_id}: {', '.join(sorted(cluster_languages))}"
                    )

        except Exception as exc:  # pragma: no cover - defensive
            print(f"  Question {base_id}: ERROR {exc}")

    if not relative_scores:
        print("WARNING: No valid clusters produced; returning 0.")
        return 0.0

    avg_relative_score = sum(relative_scores) / len(relative_scores)

    print(f"\n📊 {analysis_type.upper()} SUMMARY:")
    print(
        f"   Total {'sentence-variation combinations' if analysis_type == 'Robustness' else 'sentence IDs'} analyzed: {len(relative_scores)}"
    )
    print(f"   Average relative score: {avg_relative_score:.4f}")

    # Save cluster outputs to OUTPUT_DIR/clustering/{task_type}/
    _save_cluster_outputs(
        results, cluster_assignments, language_clusters, task_type, avg_relative_score
    )

    return avg_relative_score


def cultural_diversity(items: List[float], task_type: str, **kwargs) -> float:  # pylint: disable=unused-argument
    """Compute cultural diversity/robustness metric for a specific task type.

    Args:
        items: List of per-sample metrics (ignored, using global responses)
        task_type: Either "unspecific" (diversity) or "specific" (robustness)
    """
    if task_type not in ["unspecific", "specific"]:
        raise ValueError(
            f"Invalid task_type: {task_type}. Must be 'unspecific' or 'specific'"
        )

    return _compute_metric(task_type)


@register_aggregation("cultural_diversity_agg")
def cultural_diversity_agg(items: List[float]) -> float:
    """Aggregation for unspecific (diversity) task."""
    return cultural_diversity(items, task_type="unspecific")


@register_aggregation("cultural_robustness_agg")
def cultural_robustness_agg(items: List[float]) -> float:
    """Aggregation for specific (robustness) task."""
    return cultural_diversity(items, task_type="specific")
