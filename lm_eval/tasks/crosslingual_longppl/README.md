# Cross-Lingual Key-Token LongPPL

Cross-lingual extension of the [LongPPL](https://arxiv.org/abs/2408.09004) benchmark for evaluating long-context language modeling across 24 EU languages.

## Benchmark Definition

**LongPPL** measures a model's ability to utilize long-range context by computing perplexity only on *key tokens* — tokens whose prediction benefits most from distant context. Key tokens are identified by a teacher model (Qwen2-72B) using long-short contrastive perplexity: tokens where a long-context model significantly outperforms a short-context window are marked as "key."

**Metric:** `LongPPL = exp(weighted mean NLL on key tokens)`

Token-count weighting pools all key tokens across documents before applying `exp`, treating every key token equally regardless of which document it came from. This matches the [reference implementation](https://github.com/PKU-ML/LongPPL).

## Dataset

`dzautner/longppl-24lang-medium` — 1200 documents (50 per language) across 24 EU languages:

bg, cs, da, de, el, en, es, et, fi, fr, ga, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, sk, sl, sv

Each document includes pre-computed `key_char_spans` from the Qwen2-72B teacher model.

## Required Settings

This task requires the `longppl_hf` model, which extends `HFLM` to expose per-token losses and tokenizer offset mappings.

```bash
python -m lm_eval \
    --model longppl_hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,max_length=32768,allow_truncation=true \
    --tasks crosslingual_longppl \
    --batch_size 1
```

### Model Args

| Arg | Required | Description |
|-----|----------|-------------|
| `pretrained` | Yes | HuggingFace model name or path |
| `dtype` | Recommended | `bfloat16` for GPU inference |
| `max_length` | Recommended | Context window size (default: model's native) |
| `allow_truncation` | See note | Set `true` to truncate documents exceeding `max_length`. Without this, documents that exceed the context window raise `ValueError`. **Required for this benchmark** since some documents may exceed the model's context. |
| `auto_rope_scaling` | Optional | Set `true` to apply dynamic NTK RoPE scaling for models with short native context (e.g., Poro 8K → 32K) |
| `use_fast_tokenizer` | Required `true` | Must be a fast (Rust) tokenizer for offset_mapping support. Most models use fast tokenizers by default. |
| `parallelize` | Optional | Set `True` for multi-GPU model parallelism |

> **Note on truncation:** Unlike standard HFLM `loglikelihood_rolling`, this model does NOT use windowed passes for long documents. It performs a single forward pass per document. Documents exceeding `max_length` are either truncated (with `allow_truncation=true`) or rejected. The `key_span_coverage` metric reports what fraction of key spans survived truncation.

### Single Language

```bash
python -m lm_eval \
    --model longppl_hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,max_length=32768,allow_truncation=true \
    --tasks crosslingual_longppl_fi \
    --batch_size 1
```

### With RoPE Scaling (short-context models)

```bash
python -m lm_eval \
    --model longppl_hf \
    --model_args pretrained=LumiOpen/Poro-8B,dtype=bfloat16,max_length=32768,auto_rope_scaling=true,allow_truncation=true \
    --tasks crosslingual_longppl \
    --batch_size 1
```

## Token-Mapping Policy

Character-level key spans from the teacher are mapped to the evaluation model's token positions using **containment**: a token is marked as "key" only if its character range is *fully contained* within a key span (`tok_start >= span_start AND tok_end <= span_end`). This matches the [LongPPL reference implementation](https://github.com/PKU-ML/LongPPL/blob/main/longppl/longppl.py#L122-L137).

### Scoring Alignment

The reference implementation uses `i-1` index shifting because their loss array is offset by one position from the token array. Our implementation instead prepends `prefix_token_id` before the document tokens, which naturally aligns `losses[i]` with `offset_mapping[i]` — no index shifting needed. Both approaches score every document token including the first. This is not byte-for-byte identical to the reference, but produces the same token scoring in practice (first-token key incidence is effectively zero in this dataset).

## Reported Metrics

| Metric | Description |
|--------|-------------|
| `longppl` | Key-token LongPPL: `exp(weighted mean NLL on key tokens)` |
| `mean_key_loss` | Mean NLL on key tokens (before exp) |
| `key_span_coverage` | Fraction of key spans retained after truncation (1.0 = no loss) |
| `word_perplexity` | Standard word-level perplexity on scored text (denominator clamped to 1 if nothing scored) |
| `byte_perplexity` | Byte-level perplexity on scored text (denominator clamped to 1 if nothing scored) |
| `bits_per_byte` | Bits per byte on scored text (denominator clamped to 1 if nothing scored) |

### Per-Language vs Group LongPPL

**Per-language** (subtask) LongPPL uses the paper's token-count-weighted formula: `exp(sum(n_d * L_d) / sum(n_d))`, pooling all key tokens across that language's documents.

**Group-level** `crosslingual_longppl` LongPPL is the unweighted mean of per-language LongPPL scores (each language contributes equally regardless of key-token count). This is a cross-lingual summary statistic, not the paper's global pooled formula.

If a language subtask produces NaN LongPPL (e.g., all key tokens lost to truncation), the group-level LongPPL will also be NaN. This is intentional — a NaN group score signals that the context window is too short for at least one language.

### Interpreting `key_span_coverage`

When a document is truncated to fit the model's context window, key spans beyond the truncation point are lost. `key_span_coverage` reports the fraction of key spans fully within the scored region. Values significantly below 1.0 indicate that the context window is too short for meaningful LongPPL measurement on that language/dataset.

## Cache Caveats

The `longppl_hf` model returns `RichLoglikelihood` objects (float subclass with `.losses` and `.offset_mapping` attributes). Do not share `--use_cache` databases between `longppl_hf` and standard `hf` model runs — if a cached result is a plain float (missing per-token data), `process_results` will raise. To resolve: omit `--use_cache` or clear the cache DB.

## Citation

```bibtex
@article{wu2024longppl,
  title={LongPPL: Understanding the Long-Range Dependency of LLMs through the Lens of Key Token Loss},
  author={Wu, Yuxiang and Hu, Yuxuan and Li, Ang and Guo, Zhongxiang and Gao, Jian},
  journal={arXiv preprint arXiv:2408.09004},
  year={2024}
}
```

Note: This is a cross-lingual extension dataset (24 EU languages), not the original LongPPL benchmark dataset mix.
