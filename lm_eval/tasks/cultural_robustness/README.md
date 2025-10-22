# Cultural Robustness Evaluation

Evaluates cultural diversity and robustness in multilingual LLMs across 12 European languages.

Based on the paper "Measuring the Cultural Capabilities of LLMs across European Languages" (LREC-COLING 2024).

Dataset: [dzautner/cultural-robustness](https://huggingface.co/datasets/dzautner/cultural-robustness)

## What it does

Tests how models handle cultural questions across different languages. Two metrics:

- **Diversity**: How varied are responses when you ask the same question in different languages?
  - Example: "What should I serve my kid for breakfast?" in English vs Finnish vs Greek
  - Higher score = more cultural variation (good)

- **Robustness**: How consistent are responses when you specify the cultural context?
  - Example: "I live in Spain and want to eat like locals" asked in all languages
  - Higher score = more consistency across languages (good)

## How to run

```bash
lm_eval --model hf \
    --model_args pretrained=your-model \
    --tasks cultural_robustness \
    --batch_size 8
```

Run just diversity or robustness:
```bash
--tasks cultural_robustness_unspecific  # diversity only
--tasks cultural_robustness_specific    # robustness only
```

## Supported languages

Danish, German, Greek, English, Spanish, Finnish, Hebrew, Italian, Polish, Russian, Slovak, Swedish (12 languages, 52,200 examples)

The task auto-detects which languages your model supports from its HuggingFace model card and only tests those.

To override auto-detection, set `EVAL_LANGUAGES` environment variable:
```bash
export EVAL_LANGUAGES="english,german,spanish"
```

## How it works

1. Generates responses for cultural questions in each supported language
2. Embeds responses using sentence transformers (Qwen3-Embedding-8B)
3. Clusters responses using k-means with k ranging from 2 to n-1 languages
4. Picks best k using silhouette score
5. Computes normalized score: (k-2)/(n-3)
   - Diversity uses this directly (more clusters = more diverse)
   - Robustness inverts it: 1-(k-2)/(n-3) (fewer clusters = more robust)

## Output

Prints aggregated scores to stdout.

Optionally saves detailed clustering results to `OUTPUT_DIR/clustering/` if the `OUTPUT_DIR` environment variable is set:
- `diversity_results.csv` / `robustness_results.csv` - per-question clustering stats
- `diversity_cluster_assignments.csv` / `robustness_cluster_assignments.csv` - which language went to which cluster
- `diversity_language_clusters.csv` / `robustness_language_clusters.csv` - which languages clustered together
- `diversity_summary.json` / `robustness_summary.json` - overall scores

Note: These files are optional and not required for the final score.

## Requirements

```bash
pip install lm_eval[cultural_robustness]
```
