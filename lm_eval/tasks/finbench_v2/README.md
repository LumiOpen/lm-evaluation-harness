# 🇫🇮 FIN-bench-v2

### Paper

arXiv preprint: [https://arxiv.org/abs/2512.13330](https://arxiv.org/abs/2512.13330)

**Overview**

🇫🇮 FIN-bench-v2 is a multi-task Finnish language understanding and generation evaluation benchmark. It adapts several well-known English benchmarks and includes tasks designed to test various capabilities of language models in Finnish. The primary goal of FIN-bench-v2 is to provide a comprehensive and challenging suite for evaluating models on the Finnish language, covering a diverse set of tasks from commonsense reasoning and world knowledge to truthfulness and instruction following.

Our main evaluation principles are:

*   📊 **Task diversity**: Coverage of various task types, including multiple-choice, generative question answering, and sentence completion, to create a holistic evaluation of model capabilities.
*   🧠 **Cognitive depth**: A focus on tasks requiring reasoning, knowledge, and safety-awareness, pushing the boundaries of current model performance.
*   🇫🇮 **Finnish-centric evaluation**: Adapting and creating tasks that are linguistically and culturally relevant for the Finnish language.
*   👩🏻‍🔬 **Standardized evaluation**: Integration into the LM Evaluation Harness for flexible and reproducible evaluation.

### Tasks

| Name                                                                    | Formulation | Finnish task names                                  | Creation method | Task type                          | Task category                 | Our dataset version                                                                                                       |
|:------------------------------------------------------------------------|-------------|:----------------------------------------------------|:----------------|:-----------------------------------|:------------------------------|:--------------------------------------------------------------------------------------------------------------------------|
| [ARC-challenge-fi](https://huggingface.co/datasets/silogen/ARC-C-fi-HT) | _mcf_       | `arc_challenge_fi_mcf_fbv2_p[0-4]`                  | HT              | Multiple-choice question answering | World knowledge               | [finbenchv2-arc-c-fi-ht](https://huggingface.co/datasets/TurkuNLP/finbenchv2-arc-c-fi-ht)                                 |
|                                                                         | _cf_        | `arc_challenge_fi_cf_fbv2_p[0-4]`                   |                 |                                    |                               |                                                                                                                           |
| [Belebele](https://huggingface.co/datasets/facebook/belebele)           | _mcf_       | `belebele_fin_Latn_mcf_fbv2_p[0-4]`                 | HT              | Multiple-choice question answering | Machine reading comprehension | [finbenchv2-belebele-fi-og](https://huggingface.co/datasets/TurkuNLP/finbenchv2-belebele-fi-og)                           |
|                                                                         | _cf_        | `belebele_fin_Latn_cf_fbv2_p[0-4]`                  |                 |                                    |                               |                                                                                                                           |
| [GoldenSwag](https://huggingface.co/datasets/PleIAs/GoldenSwag)         | _mcf_       | `goldenswag_ht_fi_mcf_fbv2_p[0-4]`                  | MT&HR           | Sentence completion                | Commonsense reasoning         | [finbenchv2-goldenswag-fi-ht](https://huggingface.co/datasets/TurkuNLP/finbenchv2-goldenswag-fi-ht)                       |
|                                                                         | _cf_        | `goldenswag_ht_fi_cf_fbv2_p[0-4]`                   |                 |                                    |                               |                                                                                                                           |
| [TruthfulQA](https://huggingface.co/datasets/Eurolingua/truthfulqax)    | _cf_        | `ogx_truthfulqax_mc1_fi_fbv2_p[0-4]`                | MT              | Multiple-choice question answering | Truthfulness                  | [finbenchv2-opengpt-x_truthfulqax-fi-mt](https://huggingface.co/datasets/TurkuNLP/finbenchv2-opengpt-x_truthfulqax-fi-mt) |
|                                                                         | _cf_        | `ogx_truthfulqax_mc2_fi_fbv2_p[0-4]`                |                 |                                    |                               |                                                                                                                           |
|                                                                         | _gen_       | `ogx_truthfulqax_gen_fi_fbv2_p[0-4]`                |                 | Generative question answering      | Truthfulness                  |                                                                                                                           |
| [ScandiSent](https://github.com/timpal0l/ScandiSent)                    | _mcf_       | `scandisent_fi_mcf_fbv2_p[0-4]`                     | HW              | Multiple-choice classification     | Sentiment analysis            | [finbenchv2-scandisent-fi-mini](https://huggingface.co/datasets/TurkuNLP/finbenchv2-scandisent-fi-mini)                   |                                                                                                                           |
|                                                                         | _cf_        | `scandisent_fi_cf_fbv2_p[0-4]`                      |                 |                                    |                               |                                                                                                                           |
| [SQuAD FI](https://huggingface.co/datasets/rajpurkar/squad_v2)          | _gen_       | `squad_fi_gen_fbv2_p[0-4]`                          | MT              | Generative question answering      | Machine reading comprehension | [finbenchv2-squad-strip-fi-mt](https://huggingface.co/datasets/TurkuNLP/finbenchv2-squad-strip-fi-mt)                     |
| [SIB-200](https://huggingface.co/datasets/Davlan/sib200)                | _mcf_       | `sib200_fi_mcf_fbv2_p[0-4]`                         | HT              | Multiple-choice classification     | Text classification           | [finbenchv2-sib-200-fi-og](https://huggingface.co/datasets/TurkuNLP/finbenchv2-sib-200-fi-og)                             |
|                                                                         | _cf_        | `sib200_fi_cf_fbv2_p[0-4]`                          |                 |                                    |                               |                                                                                                                           |
| [FIN-bench](https://github.com/TurkuNLP/FIN-bench)                      | _mcf_       | `finbench_analogies_mcf_fbv2_p[0-4]`                | MT&HR           | Multiple-choice                    | Relational reasoning          | [finbenchv2-fbv1-stripped-fi-ht](https://huggingface.co/datasets/TurkuNLP/finbenchv2-fbv1-stripped-fi-ht)                 |
|                                                                         | _cf_        | `finbench_analogies_cf_fbv2_p[0-4]`                 |                 |                                    |                               |                                                                                                                           |
|                                                                         | _mcf_       | `finbench_emotions_mcf_fbv2_p[0-4]`                 | MT&HR           | Multiple-choice                    | Sentiment analysis            | [finbenchv2-fbv1-stripped-fi-ht](https://huggingface.co/datasets/TurkuNLP/finbenchv2-fbv1-stripped-fi-ht)                 |
|                                                                         | _cf_        | `finbench_emotions_cf_fbv2_p[0-4]`                  |                 |                                    |                               |                                                                                                                           |
|                                                                         | _mcf_       | `finbench_general_knowledge_mcf_fbv2_p[0-4]`        | MT&HR           | Multiple-choice                    | World knowledge               | [finbenchv2-fbv1-stripped-fi-ht](https://huggingface.co/datasets/TurkuNLP/finbenchv2-fbv1-stripped-fi-ht)                 |
|                                                                         | _cf_        | `finbench_general_knowledge_cf_fbv2_p[0-4]`         |                 |                                    |                               |                                                                                                                           |
|                                                                         | _mcf_       | `finbench_hhh_alignment_mcf_fbv2_p[0-4]`            | MT&HR           | Multiple-choice                    | Alignment and safety          | [finbenchv2-fbv1-stripped-fi-ht](https://huggingface.co/datasets/TurkuNLP/finbenchv2-fbv1-stripped-fi-ht)                 |
|                                                                         | _cf_        | `finbench_hhh_alignment_cf_fbv2_p[0-4]`             |                 |                                    |                               |                                                                                                                           |
|                                                                         | _mcf_       | `finbench_similarities_abstraction_mcf_fbv2_p[0-4]` | MT&HR           | Multiple-choice                    | Commonsense reasoning         | [finbenchv2-fbv1-stripped-fi-ht](https://huggingface.co/datasets/TurkuNLP/finbenchv2-fbv1-stripped-fi-ht)                 |
|                                                                         | _cf_        | `finbench_similarities_abstraction_cf_fbv2_p[0-4]`  |                 |                                    |                               |                                                                                                                           |

<details open>
<summary><b>Table description</b></summary>

*   **Name**: The original dataset name with a HuggingFace link, where available.
*   **Formatting**: The prompt formatting used in the task.
*   **Finnish task name**: The LM Evaluation Harness task name for the Finnish dataset.
*   **Creation method**: How the Finnish dataset was created:
    *   HW = Human-written.
    *   HT = Human translation.
    *   HR = Human review and correction of machine-translated data.
    *   MT = Machine translation of the original English dataset.
*   **Task type**: The specific type of the task.
*   **Task category**: A broader categorization of the task.
*   **Our dataset version**: Direct link to the dataset used in FIN-bench-v2.

</details>

##### Comments on specific tasks

*   **TruthfulQA**: This benchmark is split into three parts: two multiple-choice variants (`_mc1`, `_mc2`) which differ in their prompting format, and one generative variant (`_gen`).
*   **FIN-bench**: These tasks originate from the first version of FIN-bench and are all multiple-choice tasks covering a wide range of linguistic and knowledge-based challenges.

### Citation

Please use the following BibTeX entry to cite the FIN-bench-v2 paper:

```
@misc{kytöniemi2025finbenchv2unifiedrobustbenchmark,
      title={FIN-bench-v2: A Unified and Robust Benchmark Suite for Evaluating Finnish Large Language Models}, 
      author={Joona Kytöniemi and Jousia Piha and Akseli Reunamo and Fedor Vitiugin and Farrokh Mehryary and Sampo Pyysalo},
      year={2025},
      eprint={2512.13330},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.13330}, 
}
```
