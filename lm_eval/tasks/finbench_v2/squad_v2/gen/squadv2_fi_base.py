from functools import partial
from math import exp
import datasets
from packaging import version
from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask


def _squad_metric(predictions, references):
    import evaluate
    squad_metric = evaluate.load("squad_v2")
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)
    return _squad_metric(predictions=predictions, references=references).get(key, 0)


class SQuAD2FinnishBase(ConfigurableTask):
    VERSION = 1
    DATASET_PATH = "TurkuNLP/squad_v2_fi_hf"
    DATASET_NAME = None

    def __init__(self, config=None):
        # Set default prompt template - will be overridden by subclasses
        self.prompt_template = 1
        super().__init__(config=config or {"metadata": {"version": self.VERSION}})

    assert version.parse(datasets.__version__) >= version.parse("1.11.0"), (
        "datasets v1.11.0 or later required for SQuAD"
    )

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        """Multiple prompt templates for Finnish SQuAD v2"""
        prompts = {
            1: f"Otsikko: {doc['title']}\n\nTeksti: {doc['context']}\n\nKysymys: {doc['question']}\nVastaus:",
            2: f"Vastaa kysymykseen seuraavan tekstin perusteella.\nAihe: {doc['title']}\nTeksti: {doc['context']}\n\nKysymys: {doc['question']}\nVastauksesi:",
            3: f"Lue seuraava teksti ja vastaa kysymykseen. Aihe: {doc['title']}\nTeksti: {doc['context']}\nKysymys: {doc['question']}\nVastaus:",
            4: f"Tässä on teksti aiheesta: \"{doc['title']}\":\n{doc['context']}\n\nVastaa seuraavaan kysymykseen tekstin perusteella: {doc['question']}\nVastaus:",
            5: f"Aineisto:\n\n{doc['title']}\n{doc['context']}\n\nVastaa aineiston perusteella kysymykseen: \"{doc['question']}\"\n\nVastaus:"
        }
        return prompts.get(self.prompt_template, prompts[1])

    def doc_to_target(self, doc):
        answer_list = doc["answers"]["text"]
        if len(answer_list) > 0:
            answer = answer_list[0]
        else:
            answer = "Ei vastausta"
        return " " + answer

    def construct_requests(self, doc, ctx, **kwargs):
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": ["\n"], "max_new_tokens": 64}),
                idx=0,
                **kwargs,
            ),
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, " Ei vastausta"),
                idx=0,
                **kwargs,
            ),
        ]

    def process_results(self, doc, results):
        continuation, (logprob_unanswerable, _) = results
        no_answer_probability = exp(logprob_unanswerable)

        predictions = {
            "id": doc["id"],
            "prediction_text": continuation.strip(),
            "no_answer_probability": no_answer_probability,
        }

        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }

        return {
            "exact": (predictions, references),
            "f1": (predictions, references),
            "HasAns_exact": (predictions, references),
            "HasAns_f1": (predictions, references),
            "NoAns_exact": (predictions, references),
            "NoAns_f1": (predictions, references),
            "best_exact": (predictions, references),
            "best_f1": (predictions, references),
        }

    def aggregation(self):
        return {
            "exact": partial(_squad_agg, "exact"),
            "f1": partial(_squad_agg, "f1"),
            "HasAns_exact": partial(_squad_agg, "HasAns_exact"),
            "HasAns_f1": partial(_squad_agg, "HasAns_f1"),
            "NoAns_exact": partial(_squad_agg, "NoAns_exact"),
            "NoAns_f1": partial(_squad_agg, "NoAns_f1"),
            "best_exact": partial(_squad_agg, "best_exact"),
            "best_f1": partial(_squad_agg, "best_f1"),
        }

    def higher_is_better(self):
        return {
            "exact": True,
            "f1": True,
            "HasAns_exact": True,
            "HasAns_f1": True,
            "NoAns_exact": True,
            "NoAns_f1": True,
            "best_exact": True,
            "best_f1": True,
        }


# Create subclasses for each prompt variant
class SQuAD2FinnishP1(SQuAD2FinnishBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.prompt_template = 1

class SQuAD2FinnishP2(SQuAD2FinnishBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.prompt_template = 2

class SQuAD2FinnishP3(SQuAD2FinnishBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.prompt_template = 3

class SQuAD2FinnishP4(SQuAD2FinnishBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.prompt_template = 4

class SQuAD2FinnishP5(SQuAD2FinnishBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.prompt_template = 5