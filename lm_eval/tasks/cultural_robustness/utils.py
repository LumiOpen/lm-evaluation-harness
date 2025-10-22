"""Cultural robustness task utilities."""

import os
from pathlib import Path
from typing import Set

from datasets import DatasetDict, load_dataset
from huggingface_hub import model_info

from lm_eval.tasks.cultural_robustness import metrics


_DATASET_NAME_ENV = "CULTURAL_ROBUSTNESS_DATASET"
_DEFAULT_DATASET_NAME = "dzautner/cultural-robustness"

# ISO code → language name (single source of truth)
ISO_TO_LANGUAGE = {
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "fi": "finnish",
    "he": "hebrew",
    "it": "italian",
    "pl": "polish",
    "ru": "russian",
    "sk": "slovak",
    "sv": "swedish",
}

# Derive reverse mapping
LANGUAGE_TO_ISO = {lang: iso for iso, lang in ISO_TO_LANGUAGE.items()}

# Available languages (languages we have data for)
AVAILABLE_LANGUAGES = set(ISO_TO_LANGUAGE.values())

# Build language map with aliases
LANGUAGE_MAP = {}
for iso, lang in ISO_TO_LANGUAGE.items():
    LANGUAGE_MAP[iso] = lang
    LANGUAGE_MAP[lang] = lang

# Add common aliases
LANGUAGE_MAP.update(
    {
        "deutsch": "german",
        "español": "spanish",
        "italiano": "italian",
        "français": "french",
        "português": "portuguese",
        "suomi": "finnish",
    }
)

_SUPPORTED_LANGUAGES_CACHE = None


def get_model_supported_languages(model_name: str) -> Set[str]:
    """Get supported languages from HF model card, mapped to our file prefixes."""
    global _SUPPORTED_LANGUAGES_CACHE

    if _SUPPORTED_LANGUAGES_CACHE is not None:
        return _SUPPORTED_LANGUAGES_CACHE

    try:
        info = model_info(model_name)

        # Get languages from model card
        hf_languages = []
        if hasattr(info, "cardData") and info.cardData:
            if hasattr(info.cardData, "language"):
                lang = info.cardData.language
                hf_languages = lang if isinstance(lang, list) else [lang]

        # Map to our file prefixes
        supported = set()
        for lang in hf_languages:
            if lang is None:
                continue
            lang_lower = lang.lower().strip()
            if lang_lower in LANGUAGE_MAP:
                supported.add(LANGUAGE_MAP[lang_lower])

        # Intersect with available languages
        result = supported & AVAILABLE_LANGUAGES

        if result:
            print(
                f"🌐 Model {model_name} supports {len(result)} languages: {sorted(result)}"
            )
        else:
            print(
                f"⚠️  Could not determine supported languages for {model_name}, using all available"
            )
            result = AVAILABLE_LANGUAGES

        _SUPPORTED_LANGUAGES_CACHE = result
        return result

    except Exception as e:
        print(f"⚠️  Error checking model languages: {e}. Using all available languages.")
        _SUPPORTED_LANGUAGES_CACHE = AVAILABLE_LANGUAGES
        return AVAILABLE_LANGUAGES


def get_languages_to_evaluate() -> Set[str]:
    """Determine which languages to evaluate based on priority:
    1. Explicit override via EVAL_LANGUAGES env var
    2. Model card detection
    3. All available languages (fallback)
    """

    # Priority 1: Explicit override
    explicit_languages = os.environ.get("EVAL_LANGUAGES")
    if explicit_languages:
        langs = set(lang.strip().lower() for lang in explicit_languages.split(","))
        valid_langs = langs & AVAILABLE_LANGUAGES
        invalid_langs = langs - AVAILABLE_LANGUAGES

        if invalid_langs:
            print(f"⚠️  Invalid languages specified: {invalid_langs}")
            print(f"   Available languages: {sorted(AVAILABLE_LANGUAGES)}")

        if valid_langs:
            print(f"🎯 Using explicitly specified languages: {sorted(valid_langs)}")
            return valid_langs
        else:
            print("❌ No valid languages specified in EVAL_LANGUAGES")
            print(f"   Available: {sorted(AVAILABLE_LANGUAGES)}")
            raise ValueError("No valid languages to evaluate")

    # Priority 2: Model card detection
    model_name = os.environ.get("MODEL_ID") or os.environ.get("MODEL_NAME")

    if not model_name or model_name == "dummy":
        print(
            "ℹ️  No model specified or using dummy model, using all available languages"
        )
        return AVAILABLE_LANGUAGES

    # Normalize model name (handle paths like /project/cache/models/org-Model)
    if model_name.startswith("/"):
        # It's a filesystem path, extract org/model from last two components
        model_name = "/".join(model_name.rstrip("/").split("/")[-2:])

    return get_model_supported_languages(model_name)


def _get_dataset_name() -> str:
    """Get dataset name from environment or use default."""
    custom = os.environ.get(_DATASET_NAME_ENV)
    if custom:
        return custom
    return _DEFAULT_DATASET_NAME


def _load_dataset_by_type(task_type: str):
    """Load dataset from HuggingFace and filter by task type and languages."""
    metrics.reset_state()
    dataset_name = _get_dataset_name()
    languages_to_eval = get_languages_to_evaluate()

    print(f"📊 Loading dataset from {dataset_name}...")

    # Load full dataset from HuggingFace
    dataset = load_dataset(dataset_name, split="train")

    # Filter by task type
    dataset = dataset.filter(lambda x: x["type"] == task_type)

    # Filter by supported languages
    iso_codes_to_include = {LANGUAGE_TO_ISO[lang] for lang in languages_to_eval}
    dataset = dataset.filter(lambda x: x["language"] in iso_codes_to_include)

    if len(dataset) == 0:
        raise ValueError(
            f"No examples found for task_type={task_type} and languages={sorted(languages_to_eval)}"
        )

    # Get unique languages in filtered dataset
    unique_languages = sorted(set(dataset["language"]))
    print(
        f"📊 Loaded {len(dataset)} examples for {task_type} task across {len(unique_languages)} language(s): {unique_languages}"
    )

    # Return as DatasetDict with train split (required by lm-eval)
    return DatasetDict({"train": dataset})


def load_dataset_multilingual(**_: dict):
    """Load unspecific (diversity) dataset."""
    return _load_dataset_by_type("unspecific")


def load_dataset_specific(**_: dict):
    """Load specific (robustness) dataset."""
    return _load_dataset_by_type("specific")


def doc_to_text(doc):
    return doc["prompt"]


def process_results(doc, results):
    response = results[0] if isinstance(results, list) else results
    metrics.record_response(
        {
            "response": response,
            "base_id": doc["base_id"],
            "language": doc["language"],
            "type": doc["type"],
            "doc_id": doc["id"],
            "prompt": doc["prompt"],
        }
    )
    # lm-eval requires dict with metric name key, value ignored (aggregation uses global state)
    return {"cultural_diversity": 1.0}
