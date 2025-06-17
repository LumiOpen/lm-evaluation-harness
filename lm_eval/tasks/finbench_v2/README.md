# 🇫🇮 Finbench v2

### Paper & Homepage

TODO

**Overview**

🇫🇮 Finbench v2 is a multi-task Finnish language understanding and generation evaluation benchmark. It adapts several well-known English benchmarks and includes tasks designed to test various capabilities of language models in Finnish. The primary goal of Finbench v2 is to provide a comprehensive and challenging suite for evaluating models on the Finnish language, covering a diverse set of tasks from commonsense reasoning and world knowledge to truthfulness and instruction following.

Our main evaluation principles are:

*   📊 **Task diversity**: Coverage of various task types, including multiple-choice, generative question answering, and sentence completion, to create a holistic evaluation of model capabilities.
*   🧠 **Cognitive depth**: A focus on tasks requiring reasoning, knowledge, and safety-awareness, pushing the boundaries of current model performance.
*   🇫🇮 **Finnish-centric evaluation**: Adapting and creating tasks that are linguistically and culturally relevant for the Finnish language.
*   👩🏻‍🔬 **Standardized evaluation**: Integration into the LM Evaluation Harness for flexible and reproducible evaluation.

### Tasks

| Name | Finnish task name | *k*-shot | Task type | Task category | Our dataset version |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [ARC-challenge-fi](https://huggingface.co/datasets/silogen/ARC-C-fi-HT) | `arc_challenge_fi_fbv2` | ✅ | Multiple-choice question answering | World knowledge | [finbenchv2-arc-c-fi-ht](https://huggingface.co/datasets/TurkuNLP/finbenchv2-arc-c-fi-ht) |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) | `belebele_fin_Latn_cf_fbv2` | ❌ | Multiple-choice question answering | Machine reading comprehension | [finbenchv2-belebele-fi-og](https://huggingface.co/datasets/TurkuNLP/finbenchv2-belebele-fi-og) |
| | `belebele_fin_Latn_mcf_fbv2` | ❌ | Multiple-choice question answering | Machine reading comprehension | [finbenchv2-belebele-fi-og](https://huggingface.co/datasets/TurkuNLP/finbenchv2-belebele-fi-og) |
| [GoldenSwag](https://huggingface.co/datasets/PleIAs/GoldenSwag) | `finbenchv2-goldenswag-fi-ht` | ✅ | Sentence completion | Commonsense reasoning | [finbenchv2-goldenswag-fi-ht](https://huggingface.co/datasets/TurkuNLP/finbenchv2-goldenswag-fi-ht) |
| [TruthfulQA](https://huggingface.co/datasets/Eurolingua/truthfulqax) | `ogx_truthfulqax_mc1_fi_fbv2` | ❌ | Multiple-choice question answering | Truthfulness | [finbenchv2-opengpt-x_truthfulqax-fi-mt](https://huggingface.co/datasets/TurkuNLP/finbenchv2-opengpt-x_truthfulqax-fi-mt) |
| | `ogx_truthfulqax_mc2_fi_fbv2` | ❌ | Multiple-choice question answering | Truthfulness | [finbenchv2-opengpt-x_truthfulqax-fi-mt](https://huggingface.co/datasets/TurkuNLP/finbenchv2-opengpt-x_truthfulqax-fi-mt) |
| | `ogx_truthfulqax_gen_fi_fbv2`| ❌ | Generative question answering | Truthfulness | [finbenchv2-opengpt-x_truthfulqax-fi-mt](https://huggingface.co/datasets/TurkuNLP/finbenchv2-opengpt-x_truthfulqax-fi-mt) |
| [SQuAD v2](https://huggingface.co/datasets/squad_v2) | `squadv2_fi_fbv2` | ✅ | Generative question answering | Machine reading comprehension | [finbenchv2-squad_v2-fi-mt](https://huggingface.co/datasets/TurkuNLP/finbenchv2-squad_v2-fi-mt) |
| [SIB-200](https://huggingface.co/datasets/Helsinki-NLP/sib-200) | `sib200_fi_fbv2` | ✅ | Multiple-choice classification | Text classification | [finbenchv2-sib-200-fi-og](https://huggingface.co/datasets/TurkuNLP/finbenchv2-sib-200-fi-og) |
| [FIN-Bench](https://github.com/TurkuNLP/FIN-bench) | `analogies_fbv2` | ❌ | Multiple-choice | Relational reasoning | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench) |
| | `emotions_fbv2` | ❌ | Multiple-choice | Sentiment analysis | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench) |
| | `empirical_judgments_fbv2` | ❌ | Multiple-choice | Causal reasoning | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench) |
| | `general_knowledge_fbv2` | ❌ | Multiple-choice | World knowledge | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench) |
| | `hhh_alignment_fbv2` | ❌ | Multiple-choice | Alignment and safety | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench) |
| | `paraphrase_fbv2` | ❌ | Multiple-choice | Paraphrase identification | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench) |
| | `similarities_abstraction_fbv2`| ❌ | Multiple-choice | Commonsense reasoning | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench) |

<details open>
<summary><b>Table description</b></summary>

*   **Name**: The original dataset name with a HuggingFace link, where available.
*   **Finnish task name**: The LM Evaluation Harness task name for the Finnish dataset.
*   ***k*-shot**: The support for *k*-shot evaluation regimes with *k* > 0.
*   **Task type**: The specific type of the task.
*   **Task category**: A broader categorization of the task.
*   **Our dataset version**: Direct link to the dataset used in Finbench v2.

</details>

##### Comments on specific tasks

*   **Belebele**: Two variants are provided. `belebele_fin_Latn_cf_fbv2` uses a prompt format where the answer choices are shown before the question ("choices-first"), while `belebele_fin_Latn_mcf_fbv2` uses a "multiple-choices-first" template. This allows for testing prompt sensitivity.
*   **TruthfulQA**: This benchmark is split into three parts: two multiple-choice variants (`_mc1`, `_mc2`) which differ in their prompting format, and one generative variant (`_gen`).
*   **FIN-bench**: These tasks originate from the first version of FIN-bench and are all multiple-choice tasks covering a wide range of linguistic and knowledge-based challenges.

### Citation

As there is no formal paper, please cite the repository directly if you use FinBench v2 in your work:

```
@misc{finbenchv2,
  author       = {[TODO: List of authors/contributors]},
  title        = {FinBench v2: A Finnish Language Understanding and Generation Evaluation Benchmark},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[TODO: Add full URL to the GitHub directory]}}
}
```
