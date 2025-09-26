# HELMET Usage Guide

## Overview

The HELMET (How to Evaluate Long-context Language Models Effectively and Thoroughly) benchmark is now available in lm-evaluation-harness with full streaming support and dataset limiting capabilities.

## Quick Start

### Basic Usage

```bash
# Run all HELMET tasks
python -m lm_eval --model hf --model_args pretrained=your-model --tasks helmet

# Run specific HELMET task
python -m lm_eval --model hf --model_args pretrained=your-model --tasks helmet_recall

# Run multiple specific tasks
python -m lm_eval --model hf --model_args pretrained=your-model --tasks helmet_recall,helmet_rag,helmet_cite
```

### Fast Testing with HELMET_LIMIT

The HELMET dataset is very large. For faster testing, use the `HELMET_LIMIT` environment variable:

```bash
# Quick test with just 10 samples per task
HELMET_LIMIT=10 python -m lm_eval --model hf --model_args pretrained=your-model --tasks helmet_recall

# Medium test with 100 samples
HELMET_LIMIT=100 python -m lm_eval --model hf --model_args pretrained=your-model --tasks helmet

# Quick validation with 5 samples
HELMET_LIMIT=5 python -m lm_eval --model hf --model_args pretrained=your-model --tasks helmet_rag,helmet_cite
```

## Available HELMET Tasks

1. **helmet_recall** - Information retrieval from long contexts
2. **helmet_rag** - Question answering with retrieved passages
3. **helmet_rerank** - Passage ranking by relevance
4. **helmet_cite** - Citation generation
5. **helmet_longqa** - Long-form question answering
6. **helmet_summ** - Text summarization
7. **helmet_icl** - In-context learning tasks

## Features

### ✅ Streaming Dataset Support
- Uses `streaming: true` for memory-efficient processing
- Handles large datasets without loading everything into memory
- Custom `HELMETTask` class for streaming compatibility

### ✅ Zero Core Modifications
- No changes to lm_eval core modules
- All functionality isolated in `lm_eval/tasks/helmet/`
- Clean integration with existing lm_eval architecture

### ✅ Comprehensive Metrics
- **exact_match**: Exact string matching
- **f1**: Token-level F1 score
- **bleu**: BLEU score for generation tasks
- **rouge1/rouge2/rougeL**: ROUGE scores for summarization
- **bertscore**: Semantic similarity scoring
- **acc**: Accuracy for classification tasks

### ✅ Aggregation Fix
- Resolves `TypeError: unsupported operand type(s) for +: 'int' and 'list'`
- Proper scalar metric returns for aggregation
- Tested end-to-end evaluation pipeline

## Example Results

```
| Task         | Version | Filter | n-shot | Metric      | Value  |
|--------------|---------|--------|--------|-------------|--------|
| helmet_recall| 1       | none   | 0      | exact_match | 0.2340 |
|              |         |        |        | f1          | 0.4567 |
| helmet_rag   | 1       | none   | 0      | exact_match | 0.1890 |
|              |         |        |        | f1          | 0.3421 |
|              |         |        |        | bleu        | 0.2156 |
```

## Performance Tips

1. **Use HELMET_LIMIT for development**: Start with `HELMET_LIMIT=10` for quick iteration
2. **Combine with --limit**: Use both for double limiting: `HELMET_LIMIT=100 python -m lm_eval ... --limit 50`
3. **Choose specific tasks**: Don't run all 7 tasks unless needed
4. **Monitor memory**: Streaming helps but very large models may still need memory management

## Implementation Details

- **Minimal code**: Only 50+ lines in `utils.py` and `task.py`
- **Built-in metrics**: Leverages lm_eval's existing metric system
- **Trust remote code**: Automatically handles `trust_remote_code=True` for HELMET dataset
- **Error handling**: Graceful fallbacks for invalid limits or network issues

## Troubleshooting

### Dataset Download Issues
```bash
# Set CA bundle if needed
export REQUESTS_CA_BUNDLE=/path/to/ca-bundle.pem
export CURL_CA_BUNDLE=/path/to/ca-bundle.pem
```

### Memory Issues
```bash
# Use smaller limits and specific tasks
HELMET_LIMIT=50 python -m lm_eval --tasks helmet_recall --batch_size 1
```

### Network Timeouts
```bash
# The dataset is large - be patient or use smaller limits
HELMET_LIMIT=10 python -m lm_eval --tasks helmet_recall
```