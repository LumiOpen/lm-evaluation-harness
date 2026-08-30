"""
Cross-Lingual LongPPL — lm-eval process_results and aggregation functions.

These are referenced from the YAML configs via !function directives.
No task class needed — ConfigurableTask handles everything via YAML.

The model (longppl_hf) returns RichLoglikelihood values: floats that
also carry .losses and .offset_mapping attributes.  If these are
missing (wrong model), process_results raises immediately.
"""

import logging

import numpy as np

from lm_eval.tasks.crosslingual_longppl.utils import cal_overlap


eval_logger = logging.getLogger(__name__)


def process_results(doc, results):
    """
    Compute key-token LongPPL from per-token losses on the model's output.

    The longppl_hf model returns RichLoglikelihood values carrying
    .losses (per-token NLL) and .offset_mapping (char->token mapping).

    Raises RuntimeError if per-token data is missing — this indicates
    the wrong --model was used.  Use ``--model longppl_hf``.
    """
    (loglikelihood,) = results
    text = doc.get("text", "")
    key_char_spans = doc.get("key_char_spans", [])

    # Extract per-token data — must be present on every result.
    losses = getattr(loglikelihood, "losses", None)
    offset_mapping = getattr(loglikelihood, "offset_mapping", None)

    if losses is None or offset_mapping is None:
        raise RuntimeError(
            "crosslingual_longppl requires per-token loss data but "
            "the loglikelihood result has no .losses or "
            ".offset_mapping attributes.  Use --model longppl_hf "
            "(and omit --use_cache or clear cache DB if resuming)."
        )

    # Determine the actually-scored text range from offset_mapping,
    # so perplexity denominators stay correct under truncation.
    # When no tokens were scored (empty offset_mapping), scored_text
    # is empty — never fall back to full text.
    scored_end = max(end for _, end in offset_mapping) if offset_mapping else 0
    scored_text = text[:scored_end]

    # Compute key-token LongPPL
    mean_key_loss = float("nan")
    n_key_tokens = 0

    if key_char_spans:
        key_positions = cal_overlap(offset_mapping, key_char_spans)
        if key_positions:
            key_losses = [losses[p] for p in key_positions if 0 <= p < len(losses)]
            if key_losses:
                mean_key_loss = float(np.mean(key_losses))
                n_key_tokens = len(key_losses)

    # Coverage: fraction of key spans retained after truncation.
    # 0.0 when nothing was scored, 1.0 when no key spans exist.
    total_key_spans = len(key_char_spans)
    if total_key_spans == 0:
        key_span_coverage = 1.0
    elif scored_end == 0:
        key_span_coverage = 0.0
    else:
        scored_key_spans = sum(1 for _, e in key_char_spans if e <= scored_end)
        key_span_coverage = scored_key_spans / total_key_spans

    words = len(scored_text.split())
    bytes_ = len(scored_text.encode("utf-8"))

    # Guard against zero denominators: lm-eval's weighted_perplexity
    # aggregator does sum(ll)/sum(count) and crashes if sum(count)==0.
    # When nothing is scored, (0.0, 1) contributes exp(-0/1)=1.0
    # which is benign (only reachable with degenerate max_length).
    safe_words = max(words, 1)
    safe_bytes = max(bytes_, 1)

    return {
        "longppl": (mean_key_loss, n_key_tokens),
        "mean_key_loss": mean_key_loss,
        "key_span_coverage": key_span_coverage,
        "word_perplexity": (float(loglikelihood), safe_words),
        "byte_perplexity": (float(loglikelihood), safe_bytes),
        "bits_per_byte": (float(loglikelihood), safe_bytes),
    }


def nanmean(items):
    """Mean ignoring NaN values, for metric aggregation."""
    valid = [x for x in items if not np.isnan(x)]
    return float(np.mean(valid)) if valid else float("nan")


def longppl_agg(items):
    """Token-count-weighted LongPPL, matching the original paper.

    Each item is a (mean_key_loss, n_key_tokens) tuple from process_results.
    We compute: exp(sum(n_d * L_d) / sum(n_d))
    This pools all key tokens across documents before applying exp,
    equivalent to treating every key token equally regardless of which
    document it came from.
    """
    total_weighted_loss = 0.0
    total_tokens = 0
    for mean_loss, n_tokens in items:
        if np.isnan(mean_loss) or n_tokens == 0:
            continue
        total_weighted_loss += mean_loss * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("nan")
    return float(np.exp(total_weighted_loss / total_tokens))
