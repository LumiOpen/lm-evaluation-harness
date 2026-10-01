# Cultural Robustness Evaluation

Measures the cultural capabilities of multilingual LLMs across 23 languages without ground-truth labels, using the Cultural Signal Purity (CSP) metric.

Based on the paper "Measuring the Cultural Capabilities of LLMs across European Languages".

Dataset: [LumiOpen/cultural-robustness](https://huggingface.co/datasets/LumiOpen/cultural-robustness)

## What it does

The same 101 everyday questions ("What to serve my kid for breakfast?") are asked in every language, in two forms:

- **Unanchored** (`cultural_robustness_unspecific_all`): no cultural context, so the model has to infer it from the language.
- **Anchored** (`cultural_robustness_specific_all`): the same questions with an explicit location ("We live in Austria"), each asked with many countries/regions.

Answers are embedded with Qwen3-Embedding-8B and compared across languages with cosine distance:

- `U_q(L1, L2)`: distance between the answers to unanchored question `q` in languages `L1` and `L2`.
- `A_q(L1, L2)`: the same for the anchored question, averaged over its anchors.

A model with cultural knowledge gives different answers when it has to infer the culture from the language (large `U`) but agrees across languages once the culture is given (small `A`). A model that is just inconsistent has large `U` and large `A` on the same questions.

## Metrics

| Task | Metric | Definition |
|---|---|---|
| `cultural_robustness_specific_all` | `cultural_signal_purity` | **CSP**, the paper's main metric. For each language pair, the Spearman correlation ρ over questions between `U_q` and `A_q`; CSP = 1 − ρ̄², where ρ̄ is the mean over language pairs. |
| `cultural_robustness_specific_all` | `cultural_signal_purity_question_level` | The earlier formulation: one Spearman correlation between per-question means of `U` and `A` over all pairs; 1 − ρ². It penalises models evaluated on more languages. |
| `cultural_robustness_unspecific_all` | `cultural_diversity` | **CD**: mean `U` over questions and language pairs. |
| `cultural_robustness_specific_all` | `cultural_robustness` | **CR**: 1 − mean `A`, averaged per question over its anchors, then over questions. |

Higher is better for all four. CSP needs both tasks in the same run; with only one of them it is reported as NaN.

## How to run

```bash
lm_eval --model vllm \
    --model_args pretrained=your-model,gpu_memory_utilization=0.6 \
    --tasks cultural_robustness \
    --apply_chat_template \
    --batch_size auto
```

Generation is greedy (`do_sample=False`) with `max_gen_toks=200`, as in the paper. `--gen_kwargs max_gen_toks=N` overrides the limit, but the scores are then not comparable with the paper's.

The embedding model is loaded on the GPU in the same process as the evaluated model after generation, so leave room for it (about 16 GB in bf16) when using vLLM.

Run it in a single lm-eval process. The score is computed from responses held in module-level state, so multi-process data parallelism (e.g. `accelerate launch` with the `hf` backend) would only score rank 0's share of the data.

## Languages

Bengali, Catalan, Czech, Danish, English, Faroese, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Kannada, Marathi, Polish, Russian, Slovak, Spanish, Swedish, Tamil, Telugu, Turkish (23 languages, 129,283 prompts).

Language selection, in priority order:

1. `EVAL_LANGUAGES`, a comma-separated list of language names:
   ```bash
   export EVAL_LANGUAGES="english,german,spanish,finnish"
   ```
2. The `language` field of the HuggingFace model card of the repo named in the `MODEL_ID` (or `MODEL_NAME`) environment variable. `--model_args pretrained=...` is not read, so without one of these variables every language is used.
3. All 23 languages.

The paper evaluates each model on the languages its model card lists.

## Configuration

Environment variables:

| Variable | Default | |
|---|---|---|
| `EVAL_LANGUAGES`, `MODEL_ID` / `MODEL_NAME` | | Language selection, see above |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | Hub id or local path of the sentence-transformers model |
| `EMBEDDING_DEVICE` | `cuda:0` if available, else `cpu` | Device for the embedding model |
| `EMBEDDING_BATCH_SIZE` | `64` | Embedding batch size |
| `CULTURAL_ROBUSTNESS_DATASET` | `LumiOpen/cultural-robustness` | Dataset to load |

## Output

The four scores appear in the standard lm-eval results. The INFO log line `cultural_robustness scores: ...` also gives ρ̄, the question-level ρ and the number of language pairs and questions used. Per-sample responses are available with `--log_samples`.

## Requirements

```bash
pip install lm_eval[cultural_robustness]
```
