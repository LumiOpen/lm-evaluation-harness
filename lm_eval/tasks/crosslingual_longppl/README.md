# Cross-Lingual LongPPL

Cross-Lingual LongPPL evaluates long-context language modeling capability on **any language** without requiring a model that understands that language.

## Method

The method transfers "key paragraph" positions from English to target languages using cross-lingual embeddings:

1. **Teacher Model** (Qwen2-72B): Computes per-paragraph perplexity on English documents, selecting the top-k highest-PPL paragraphs as "key" paragraphs
2. **Embedding Model** (Qwen3-Embedding-8B): Aligns English paragraphs to target language paragraphs using cosine similarity (threshold >= 0.5)
3. **Evaluated Model**: Computes perplexity on aligned key paragraphs in target language
4. **Metric**: Mean perplexity across key paragraphs

### Validated Configuration

| Parameter | Value |
|-----------|-------|
| Granularity | Natural paragraphs |
| Top-k | 8 paragraphs |
| Alignment Threshold | 0.5 cosine similarity |
| Teacher Model | Qwen2-72B-Instruct |
| Embedding Model | Qwen3-Embedding-8B |

### Validation Results

**Correlation:** Spearman rho = 0.978 with native evaluation

## Leaderboard

Mean perplexity (lower is better) on aligned key paragraphs across 12 languages:

| Model | EN | DE | FR | ES | FI | SV | PL | LT | HU | EL | MT | LV |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Llama-Poro-2-70B-base** | 20.88 | 7.17 | 7.68 | 9.33 | **4.25** | 6.88 | 5.67 | 6.41 | 4.38 | 5.45 | 7.47 | 6.14 |
| **Llama-Poro-2-8B-base** | 27.85 | 10.32 | 10.83 | 13.47 | **4.71** | 9.90 | 8.83 | 10.21 | 7.01 | 8.79 | 17.19 | 9.46 |
| Llama-Poro-2-70B-Instruct | 25.55 | 8.64 | 8.99 | 10.81 | 4.93 | 8.62 | 6.78 | 8.14 | 5.29 | 6.65 | 9.90 | 7.74 |
| Apertus-70B-2509 | 21.02 | 9.57 | 10.06 | 11.23 | 6.45 | 7.47 | 6.52 | **5.86** | 6.16 | 4.66 | **5.37** | **5.54** |
| Viking-33B | **8.24** | **6.77** | **7.10** | **7.28** | 6.59 | **6.70** | **5.46** | 12.06 | 5.72 | 2.89 | 17.92 | 9.36 |
| Meta-Llama-3.1-8B | 25.50 | 9.12 | 9.32 | 11.60 | 7.75 | 8.79 | 7.10 | 8.32 | 5.24 | 6.57 | 9.47 | 7.80 |
| EuroLLM-22B-Instruct | 14.56 | 11.95 | 9.17 | 11.57 | 9.06 | 9.05 | 10.03 | 8.99 | 9.27 | 5.67 | 7.58 | 9.27 |
| Mistral-7B-v0.3 | 17.28 | 8.50 | 8.36 | 9.37 | 12.03 | 9.32 | 6.65 | 16.59 | 6.06 | **2.72** | 27.60 | 15.51 |
| Gemma-3-27B-IT | 34.92 | 20.02 | 17.89 | 24.07 | 9.78 | 13.64 | 14.24 | 11.40 | 10.12 | 7.14 | 8.73 | 10.05 |
| Qwen2.5-7B | 24.31 | 10.59 | 8.99 | 10.78 | 9.83 | 12.28 | 10.81 | 10.94 | 8.66 | 2.56 | 48.57 | 8.94 |
| Falcon-7B | 22.80 | 13.21 | 9.01 | 11.13 | 22.60 | 15.20 | 12.48 | 22.29 | 14.37 | 2.79 | 23.00 | 21.29 |

**Bold** = best score in column. Lower PPL = better long-context capability.

### Key Observations

- **Finnish (FI)**: Poro models dominate (trained on Finnish data)
- **Viking-33B**: Best on Western European languages (EN, DE, FR, ES, SV, PL)
- **Greek (EL)**: Low PPL values due to tokenization (1 char/token vs 3.5 for English)
- **Maltese (MT)**: High variance - challenging low-resource language

## Tasks

| Task | Language | Family |
|------|----------|--------|
| `crosslingual_longppl_en` | English | Germanic |
| `crosslingual_longppl_de` | German | Germanic |
| `crosslingual_longppl_fr` | French | Romance |
| `crosslingual_longppl_es` | Spanish | Romance |
| `crosslingual_longppl_fi` | Finnish | Uralic |
| `crosslingual_longppl_sv` | Swedish | Germanic |
| `crosslingual_longppl_pl` | Polish | Slavic |
| `crosslingual_longppl_lt` | Lithuanian | Baltic |
| `crosslingual_longppl_hu` | Hungarian | Uralic |
| `crosslingual_longppl_el` | Greek | Hellenic |
| `crosslingual_longppl_mt` | Maltese | Semitic |
| `crosslingual_longppl_lv` | Latvian | Baltic |

## Usage

### Quick Start (Recommended)

The task automatically downloads pre-computed aligned documents from [dzautner/crosslingual-longppl-eurovoc](https://huggingface.co/datasets/dzautner/crosslingual-longppl-eurovoc):

```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B \
    --tasks crosslingual_longppl_fi,crosslingual_longppl_lt \
    --batch_size 1 \
    --num_fewshot 0
```

No setup required - it just works.

### Custom Aligned Documents (Optional)

To use your own aligned documents (e.g., different teacher model or documents):

**Step 1:** Generate aligned documents:

```bash
python -m lm_eval.tasks.crosslingual_longppl.prepare_aligned_docs \
    --teacher_model /path/to/Qwen2-72B-Instruct \
    --dataset_dir /path/to/eurovoc \
    --output aligned_docs.json \
    --n_docs 20 \
    --top_k 8
```

**Step 2:** Point to your file:

```bash
export CROSSLINGUAL_LONGPPL_ALIGNED_DOCS=/path/to/aligned_docs.json
```

**Step 3:** Run evaluation (same as above)

## Metrics

| Metric | Description |
|--------|-------------|
| `longppl` | Mean perplexity on aligned key paragraphs |
| `log2_longppl` | Log2 of mean perplexity |

Lower values indicate better long-context capability.

## Documents Used

20 parallel EU legal documents from Eurovoc:

| # | CELEX ID | # | CELEX ID |
|---|----------|---|----------|
| 1 | 32012R0771 | 11 | 32012D0421 |
| 2 | 32012D0484 | 12 | 32012D0423 |
| 3 | 32012D0452 | 13 | 32012D0443 |
| 4 | 32013D0664 | 14 | 32012D0422 |
| 5 | 32013D0009 | 15 | 32012D0395 |
| 6 | 32013D0199 | 16 | 32012R0645 |
| 7 | 32013D0283 | 17 | 32012D0409 |
| 8 | 32012D0836 | 18 | 32012R0627 |
| 9 | 32012R0692 | 19 | 32012R0918 |
| 10 | 32012D0427 | 20 | 32012D0362 |

## Known Limitations

1. **Tokenization effects**: Different scripts tokenize differently (e.g., Greek ~1 char/token vs English ~3.5). Raw PPL values differ across languages but rankings remain valid within a language.

2. **Few-shot not supported**: This is a perplexity task; `num_fewshot` must be 0.

## Citation

Based on LongPPL with cross-lingual transfer extension.

```bibtex
@article{longppl2024,
  title={LongPPL: Long-Context Language Model Evaluation via Key Token Perplexity},
  author={...},
  year={2024}
}
```
