def construct_requests(doc, ctx, **kwargs):
    return [
        {"request_type": "generate_until", "arguments": (ctx, {"until": ["\n"]})},
        {"request_type": "loglikelihood", "arguments": (ctx, " Ei vastausta")}
    ]

def _squad_metric(predictions, references):
    import evaluate
    
    squad_metric = evaluate.load("squad_v2")
    return squad_metric.compute(predictions=predictions, references=references)


def process_results(doc, results):
    from math import exp

    continuation, (logprob_unanswerable, _) = results
    no_answer_probability = exp(logprob_unanswerable)

    # Build prediction and reference structures for SQuAD metric
    predictions = {
        "id": doc["id"],
        "prediction_text": continuation.strip(),
        "no_answer_probability": no_answer_probability
    }
    
    references = {
        "id": doc["id"], 
        "answers": doc["answers"],
    }
    
    # Calculate exact match and F1 using the SQuAD metric
    squad_results = _squad_metric([predictions], [references])
    
    return {
        "exact_match": squad_results.get("exact", 0.0),
        "f1": squad_results.get("f1", 0.0)
    }