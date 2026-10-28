# Lingsoft PEMT — Finnish translation quality & fluency

### Source

Corpus: [LumiOpen/Lingsoft-EU-Summaries-PEMT](https://github.com/LumiOpen/Lingsoft-EU-Summaries-PEMT) (Jira: BASE-606)
Eval data: [LumiOpen/lingsoft-pemt-evals](https://huggingface.co/datasets/LumiOpen/lingsoft-pemt-evals)

**Overview**

Evals for translation quality and fluency built from professional post-edited
machine translations of "Summaries of EU Legislation" (2016–2025). Each item
pairs a raw MT output (`Raw_MT`) with its professional human post-edit
(`Target_PEMT`) for the same English source sentence. Only pairs where
post-editing actually changed the text are used, and the post-edited side is
always defined as the correct answer.

This is a **preliminary Finnish-only version** (built for Poro 3 evaluation).
A 23-language version from the same corpus is planned; the dataset configs
(`FI_*`) and task names (`lingsoft_pemt_fi_*`) are prefixed accordingly so
other languages can be added alongside.

### Tasks

Two formulations over the original translation-quality pairs, plus a separate
annotated fluency task for post-trained models:

| Task | Formulation | What it measures | Split | n |
|:--|:--|:--|:--|--:|
| `lingsoft_pemt_fi_mcf_shared_5shot` | _mcf_, 5-shot, logprob over `" A"`/`" B"` | blind A/B preference judgment | shared | 1,054 |
| `lingsoft_pemt_fi_mcf_broad_5shot` | _mcf_, 5-shot, logprob over `" A"`/`" B"` | blind A/B preference judgment | broad | 8,144 |
| `lingsoft_pemt_fi_bpb_shared` | _bpb_, 0-shot, unconditional | Finnish fluency/naturalness by likelihood | shared | 527 |
| `lingsoft_pemt_fi_bpb_broad` | _bpb_, 0-shot, unconditional | Finnish fluency/naturalness by likelihood | broad | 4,072 |
| `lingsoft_pemt_fi_fluency` | 0-shot, generated A/B answer | Finnish fluency/naturalness preference | annotated fluency | 452 |

(mcf has 2× the rows of bpb because every pair appears in both A/B orders.)

The fluency task uses the separate `FI_fluency` config: 226 deduplicated,
annotated pairs, each presented in both A/B orders (452 prompts). It is not a
subset selected dynamically from the shared/broad tasks.

**fluency** — designed for post-trained models with `--apply_chat_template`.
Only the two Finnish candidates are shown. The English source, annotation
labels, and candidate provenance remain sample metadata. The prompt is:

```text
Which Finnish text is more fluent and natural? Answer with A or B only.

Finnish text A: {candidate A}
Finnish text B: {candidate B}
```

There are no demonstrations. The model generates its answer; after stripping
surrounding whitespace, a single `A` or `B` (case-insensitive) is accepted.
Explanations, empty answers, and unstripped reasoning are invalid and count as
incorrect. Reasoning models should use their backend's native
`think_end_token` setting so scoring receives only the final answer. Both
candidate orders share a `pair_id`, and the post-edited candidate is the gold
answer in each order. This measures fluency preference; the English source
is not provided for judging translation accuracy.


The `shared` split contains text units from 6 segments shared across all 23
corpus languages (cross-lingually comparable once other languages are added);
`broad` is a broad sample across all segments.

**mcf** — the prompt shows the English source and both translations labeled
only "Finnish translation A/B" and asks for the preferred one; the two
single-token continuations `" A"`/`" B"` are scored by loglikelihood. Every
pair appears twice with the A/B order swapped, so random chance is exactly
0.5. The 5 few-shot examples are pre-rendered into the prompts in the dataset
(sampled from the FI train split with balanced answer labels), so the task
runs with `num_fewshot: 0` — **do not pass `--num_fewshot`**. Works for base
and instruction-tuned models.

**bpb** — ranks the two Finnish text units by unconditional loglikelihood
with byte-length normalization (bits-per-byte comparison; the texts differ in
length, so `acc_bytes` is the headline metric). No source, no prompt, no
instruction — usable on any base model.

Note bpb is deliberately **unconditional**. A source-conditioned variant
(`English: {source}\nFinnish:` + BPB of each continuation) was implemented
and evaluated, and the signal inverts: given the source, the raw MT is the
more *predictable* continuation (it is the near-literal rendering), so every
model tested scored 0.35–0.45 — the formulation measures literalness rather
than quality (see [Pistotti et al., 2025](https://aclanthology.org/2025.uncertainlp-main.7/))
and was dropped.

Why conditioning inverts the signal (working analysis): post-editors change
the raw MT for two distinct reasons — *fluency* (the MT is unnatural Finnish)
and *translation accuracy/terminology* (the MT is wrong or off the official
term base). Manual inspection of flipped pairs shows the second kind
dominates the inversion: the editor's fix imports knowledge from outside the
sentence (e.g. the official EU term-base rendering), which makes the
post-edit *less* predictable from the source alone, while the literal raw MT
becomes more predictable. Conditioning therefore rewards literalness and
penalizes exactly the edits that make the text better. A per-edit human
annotation layer separating fluency edits from accuracy/terminology edits is
in progress (Danish complete — 256 fluency / 165 accuracy segments — with
propagation to the other languages underway); once it covers Finnish we will
split the eval by edit type and quantify how much of the conditional
inversion is attributable to each. Until then, treat the unconditional bpb
and the blind mcf preference as the quality signals, and any
source-conditioned formulation as a literalness probe.

### Groups

- `lingsoft_pemt_fi_mcf_5shot` — both mcf tasks, size-weighted mean.
- `lingsoft_pemt_fi_bpb` — both bpb tasks, size-weighted mean.

### Metrics

- `acc` (all tasks): rate of preferring the human post-edit.
- `pair_acc` (mcf): fraction of pairs answered correctly in **both** A/B
  orders — the position-bias-corrected number. A model that always emits the
  same letter scores 0.5 `acc` but 0.0 `pair_acc`; unbiased random guessing
  scores 0.25. Pairs with only one order present (e.g. under `--limit`) are
  dropped. (`acc_norm` is omitted for mcf: both continuations are single
  tokens of equal length, so it equals `acc`.)
- `acc_bytes` (bpb): byte-length-normalized comparison = bits-per-byte
  ranking; headline metric. `acc_norm` (character-normalized) is reported
  alongside.
- `acc` / `pair_acc` (fluency): generated-answer accuracy and accuracy in both
  candidate orders, with the same pair interpretation and chance levels as mcf.
- `invalid_answer_rate` (fluency): fraction of responses that are not a bare
  A/B answer. Invalid answers are included in the accuracy denominator.

### How to run

```bash
lm_eval --model hf \
    --model_args pretrained=LumiOpen/Llama-Poro-2-8B-base,dtype=bfloat16 \
    --tasks lingsoft_pemt_fi_mcf_5shot,lingsoft_pemt_fi_bpb \
    --batch_size auto
```

### Running the post-trained fluency task

Use a chat template. Task defaults are greedy decoding and a 32,768-token
budget, allowing reasoning models to finish their trace. Generation settings
can be overridden through `--gen_kwargs`. For example, with a local Qwen3
checkpoint and the vLLM backend:

```bash
lm_eval --model vllm \
    --model_args 'pretrained=/path/to/Qwen3-0.6B,enable_thinking=true,think_end_token=</think>' \
    --tasks lingsoft_pemt_fi_fluency \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs 'max_gen_toks=32768,do_sample=True,temperature=0.6,top_p=0.95' \
    --batch_size auto \
    --log_samples \
    --output_path results/lingsoft_pemt_fi_fluency
```

Use `think_end_token` only for models that emit that reasoning delimiter.
For non-reasoning instruction models, omit it and `enable_thinking`. Run all
452 prompts for the complete evaluation. If limiting samples for validation,
use an even limit: candidate orders are adjacent, and incomplete pairs are
excluded from `pair_acc`.

### Fluency validation (October 2026)

Validated with vLLM 0.20.0rc1, chat templates, sampling at temperature 0.6 /
top-p 0.95, and the default harness seeds:

| Model | Prompts | Token budget | acc | pair_acc | Invalid answers |
|:--|--:|--:|--:|--:|--:|
| Qwen3-0.6B, thinking enabled, `think_end_token=</think>` | 20 (10 pairs, smoke only) | 4,096 | 0.6500 | 0.5000 | 0 |
| Qwen3-4B-Instruct-2507, non-reasoning | 452 (full set) | 32,768 | 0.5133 | 0.0575 | 0 |

Neither run truncated a response. The full run's scores were independently
recomputed from the logged final answers. It chose B in 410/452 responses;
the low `pair_acc` exposes that position bias despite near-chance `acc`.
The smoke subset and full-set results are not a model comparison. Unit tests
cover candidate swapping, gold labels, metadata exclusion, invalid answers,
and pair aggregation. Full-data validation confirmed 226 distinct changed
pairs, 452 prompts, and 226 gold labels of each letter.

### Reference results (July 2026)

mcf (`acc` / `pair_acc`, full data, logprob): Poro-2-70B-base 0.564/0.345 ·
Qwen2.5-72B 0.517/0.313 · EuroLLM-22B-Instruct 0.514/0.149 · 8B-class and
instruction-tuned variants ≈ chance. Frontier models answering the same
prompts generatively (150-pair subset) reach ≈ 0.62–0.64 acc, which is the
practical ceiling observed: even frontier models consistently prefer the raw
MT on ~a quarter of pairs (preferential/house-style edits with no objective
winner).

bpb (`acc_bytes`): all base models land above chance and the score grows with
Finnish capability — Poro-2-70B-base 0.599 · EuroLLM-22B 0.588 ·
Llama-3.1-8B 0.563 · Poro-2-8B 0.555 · Qwen2.5 (0.5B–72B) 0.526–0.545.
Useful as a graded tracking metric for base-model training.

The near-chance base-model mcf scores match the literature on MT-vs-post-edit
preference ([Berger et al., WMT 2024](https://arxiv.org/abs/2410.02320)).

### Known behavior & ongoing work

Conditioning the likelihood comparison on the English source systematically
favors the raw MT (it is the most source-predictable rendering), which is why
bpb is unconditional; this held across six models and was confirmed with
shuffled-source and document-context controls. The effect is expected to
interact with edit type: the source should help detect translation-accuracy
errors (omissions, mistranslations) while penalizing fluency/terminology
edits. Edit-type annotations (fluency vs. translation accuracy) are being
produced for the corpus; once available, we will revisit the formulation —
including a conditional variant restricted to translation-error pairs — and
get to the bottom of the issue. The original suite ships mcf (for instruction-tuned and strong base models)
and unconditional bpb (for base models). The separate annotated `FI_fluency`
task evaluates generated fluency preferences on post-trained models.

### Dataset provenance

Professional translation and post-editing by Lingsoft under "Summaries of EU
Legislation" (2016–2025). Original editorial content © European Union,
re-used via the Publications Office of the European Union under CC-BY 4.0.
Alignment and raw MT layers provided by Lingsoft. The mcf prompts are
generated by `create_fi_base_model_eval.py` in the source repo (5 shots,
seed 42); the bpb pairs are the changed rows of `splits/FI/test_*.jsonl`.
The mcf formulation matches the source repo's `score_fi_base_model_eval.py`
scoring scheme; harness results were verified against an independent
recomputation from logged samples.
