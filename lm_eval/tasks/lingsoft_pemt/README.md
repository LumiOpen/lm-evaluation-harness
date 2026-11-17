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

Two formulations over the same underlying pairs, a Finnish morphology
diagnostic, and a seven-choice Finnish fluency fill-in-the-blank set:

| Task | Formulation | What it measures | Split | n |
|:--|:--|:--|:--|--:|
| `lingsoft_pemt_fi_mcf_shared_5shot` | _mcf_, 5-shot, logprob over `" A"`/`" B"` | blind A/B preference judgment | shared | 1,054 |
| `lingsoft_pemt_fi_mcf_broad_5shot` | _mcf_, 5-shot, logprob over `" A"`/`" B"` | blind A/B preference judgment | broad | 8,144 |
| `lingsoft_pemt_fi_bpb_shared` | _bpb_, 0-shot, unconditional | Finnish fluency/naturalness by likelihood | shared | 527 |
| `lingsoft_pemt_fi_bpb_broad` | _bpb_, 0-shot, unconditional | Finnish fluency/naturalness by likelihood | broad | 4,072 |
| `lingsoft_pemt_fi_morphology_mcf` | 4-choice mcf, 0-shot, logprob over `" A"`–`" D"` | Finnish inflection choice in context | test | 400 (100 items × 4 rotations) |
| `lingsoft_pemt_fi_blank_pairwise` | 7-choice as A/B pairs, 5-shot, logprob over `" A"`/`" B"` | Finnish fluency: which fill reads most naturally | test | 9,366 (223 items × 42 ordered pairs) |
| `lingsoft_pemt_fi_blank_cf` | 7-choice cloze, 5-shot, logprob of each fill | Finnish fluency: which fill reads most naturally | test | 223 |

(mcf has 2× the rows of bpb because every pair appears in both A/B orders.)

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

**morphology mcf** — each item blanks one word of the professional post-edit
and offers four forms of the same lemma: the professional form, the attested
raw-MT form, and two Omorfi-generated forms that are valid Finnish and attested
elsewhere in the corpus. The model sees only the Finnish sentence; the four
single-token continuations `" A"`–`" D"` are scored by loglikelihood. Each
item appears in four rotations so every candidate occupies every letter once
(chance = 0.25). Works for base and instruction-tuned models without a chat
template. The items were prescreened by an assistant but have **not** been
natively adjudicated for a unique valid answer, so treat this as a development
diagnostic.

**blank pairwise / cf** — a preliminary Finnish fluency set from Maria
Barrett (config `FI_blank_7choice`; other languages follow once their review
completes). Each item shows up to two preceding sentences of a Finnish
passage, blanks the next text unit, and offers seven candidate fills: the
professional post-edit (gold) and six distractors, each the gold with one
edit. Both tasks take their five demonstrations from the dataset's separate
`fewshot` split (`first_n`), so they run **5-shot** — pass `--num_fewshot 5`
or omit the flag; `--num_fewshot 0` silently drops the demonstrations.

The pairwise variant never shows all seven fills at once. Each prompt shows
the passage and exactly two candidates labelled A and B; all 21 pairs are
asked in both orders, so each item becomes 42 prompts. The demonstrations are
the few-shot items reduced to the gold against one distractor, with the gold
alternating between A and B. Nothing is generated: a prompt's preference is
log p(`" A"`) − log p(`" B"`), the two orders of a pair are averaged (which
cancels position bias), and the candidate with the higher averaged preference
wins the pair. The headline metric `item_copeland_acc` counts each
candidate's wins over its six pairs and scores an item correct when the gold
has the most wins (ties split; chance 1/7).

The cloze variant scores each fill as a continuation of the preceding passage
(no instruction, no option list); its headline metric is `acc_norm` because
the fills differ in length. 11 test items have no preceding sentence, so their
cloze context is just the demonstrations.

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
- `lingsoft_pemt_morphology` (tag) — the morphology mcf task.
- `lingsoft_pemt_blank` (tag) — both blank tasks (`lingsoft_pemt_blank_pairwise`
  tags the pairwise task alone).

### Metrics

- `acc` (all tasks except blank pairwise): rate of preferring the human
  post-edit (morphology mcf: rate of picking the professional form, per
  rotation record; blank cf: rate of picking the professional fill, chance
  1/7).
- `pair_acc` (mcf): fraction of pairs answered correctly in **both** A/B
  orders — the position-bias-corrected number. A model that always emits the
  same letter scores 0.5 `acc` but 0.0 `pair_acc`; unbiased random guessing
  scores 0.25. Pairs with only one order present (e.g. under `--limit`) are
  dropped. (`acc_norm` is omitted for mcf: both continuations are single
  tokens of equal length, so it equals `acc`.)
- `rotation_acc` (morphology mcf): item-level accuracy after averaging each
  candidate's letter loglikelihood over its four positions, so letter-position
  bias cancels; headline metric. Chance is 0.25.
- `acc_bytes` (bpb): byte-length-normalized comparison = bits-per-byte
  ranking; headline metric. `acc_norm` (character-normalized) is reported
  alongside.
- `acc_norm` (blank cf): character-length-normalized comparison of the seven
  fills; headline metric for the cloze variant.
- Blank pairwise (all computed in `utils_pairwise.py`; "debiased" means the
  two orders of a pair are averaged):
  - `item_copeland_acc`: headline; gold has the most debiased pairwise wins
    among the seven candidates (ties split, chance 1/7).
  - `item_beats_all`: gold wins every one of its six debiased pairs.
  - `gold_pair_acc_debiased`: gold-vs-distractor pairs won after averaging
    both orders (chance 0.5).
  - `gold_pair_acc`: gold-vs-distractor pairs won in **both** orders (chance
    0.25).
  - `gold_acc`: per-prompt rate of picking the gold where it is shown (chance
    0.5).
  - `first_choice_rate`: share of prompts answered `" A"`; 0.5 means no
    position bias.

### How to run

```bash
lm_eval --model hf \
    --model_args pretrained=LumiOpen/Llama-Poro-2-8B-base,dtype=bfloat16 \
    --tasks lingsoft_pemt_fi_mcf_5shot,lingsoft_pemt_fi_bpb \
    --batch_size auto
```

The blank tasks are 5-shot by default; run them with `--tasks lingsoft_pemt_blank`
(or the two task names) and either omit `--num_fewshot` or pass `5`.

Reference blank pairwise `item_copeland_acc` (Maria Barrett): Qwen3.5-9B-Base
0.36 · Qwen3.5-9B 0.48 · Qwen3.5-35B-A3B-Base 0.54 · Qwen3.5-35B-A3B 0.49.

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
get to the bottom of the issue. Until then this suite ships mcf (for
instruction-tuned and strong base models) and unconditional bpb (for base
models).

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
