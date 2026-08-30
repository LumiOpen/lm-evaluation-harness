"""
Utility functions for Cross-Lingual LongPPL lm-eval task.

The dataset dzautner/longppl-24lang-medium contains documents with
pre-computed key_char_spans from a Qwen2-72B teacher model. These
character-level spans are mapped to the eval model's token positions
using cal_overlap(), then used to compute key-token LongPPL.
"""

LANGUAGES = [
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "ga",
    "hr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
]


def cal_overlap(
    offset_mapping: list[tuple[int, int]],
    key_char_spans: list[list[int]],
) -> list[int]:
    """
    Map character-level key spans to token indices using containment.

    A token is marked as "key" if its character range is fully contained
    within ANY key span (tok_start >= span_start AND tok_end <= span_end),
    matching the LongPPL reference implementation.  key_char_spans must
    be sorted by start position.

    Handles overlapping and nested spans correctly by scanning all
    candidate spans per token.

    Args:
        offset_mapping: List of (char_start, char_end) per token from tokenizer.
        key_char_spans: List of [char_start, char_end] spans marking key content.

    Returns:
        Token indices fully contained within at least one key span.
    """
    if not key_char_spans:
        return []

    key_tokens = []
    j_start = 0  # earliest span that could still contain future tokens

    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_end <= tok_start:
            continue

        # Advance past spans that end before this token starts —
        # they can't contain this or any later token.
        while j_start < len(key_char_spans) and key_char_spans[j_start][1] <= tok_start:
            j_start += 1

        # Check all candidate spans from j_start onward.
        # Since spans are sorted by start, once span_start > tok_start
        # no further span can contain this token.
        for j in range(j_start, len(key_char_spans)):
            span_start, span_end = key_char_spans[j]
            if span_start > tok_start:
                break
            if tok_end <= span_end:
                key_tokens.append(i)
                break

    return key_tokens


# ---------- lm-eval YAML hook functions -----------


def _make_process_docs(lang):
    """Create a language-specific process_docs function."""

    def process_docs(dataset):
        return dataset.filter(
            lambda row: row.get("language") == lang and bool(row.get("key_char_spans"))
        )

    return process_docs


# Generate per-language process_docs functions.
# Each is referenced from the per-language YAML as:
#   process_docs: !function utils.process_docs_fi
for _lang in LANGUAGES:
    globals()[f"process_docs_{_lang}"] = _make_process_docs(_lang)
