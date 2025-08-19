# Based on NorEval - https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/noreval

import datasets
import transformers.data.metrics.squad_metrics as squad_metrics


def process_results(doc, results):
    preds = results[0]
    reference = doc["answers"]["text"][0]
    f1_sum = squad_metrics.compute_f1(reference, preds)
    exact_match = squad_metrics.compute_exact(reference, preds)
    return {"f1": f1_sum, "exact_match": exact_match}


def p1(doc):
    title = doc["title"]
    passage = doc["context"]
    question = doc["question"]
    prompt = f"Otsikko: {title}\n\nTeksti: {passage}\n\nKysymys: {question}\nVastaus:"
    return prompt


def p2(doc):
    title = doc["title"]
    passage = doc["context"]
    question = doc["question"]
    prompt = f"Vastaa kysymykseen seuraavan tekstin perusteella.\nAihe: {title}\nTeksti: {passage}\n\nKysymys: {question}\nVastauksesi:"
    return prompt


def p3(doc):
    title = doc["title"]
    passage = doc["context"]
    question = doc["question"]
    prompt = (
        f"Lue seuraava teksti ja vastaa kysymykseen. Aihe: {title}\nTeksti: {passage}\nKysymys: {question}\nVastaus:"
    )
    return prompt


def p4(doc):
    title = doc["title"]
    passage = doc["context"]
    question = doc["question"]
    prompt = f'Tässä on teksti aiheesta: "{title}":\n{passage}\n\nVastaa seuraavaan kysymykseen tekstin perusteella: {question}\nVastaus:'
    return prompt


def p5(doc):
    title = doc["title"]
    passage = doc["context"]
    question = doc["question"]
    prompt = f'Aineisto:\n\n{title}\n{passage}\n\nVastaa aineiston perusteella kysymykseen: "{question}"\n\nVastaus:'
    return prompt