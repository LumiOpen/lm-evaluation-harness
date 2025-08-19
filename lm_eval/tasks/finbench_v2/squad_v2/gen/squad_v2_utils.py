def _squad_metric(predictions, references):
    import evaluate
    
    squad_metric = evaluate.load("squad_v2")
    return squad_metric.compute(predictions=predictions, references=references)

def process_results(doc, results):
    """Process results for SQuAD v2 Finnish task with any prompt variant"""
    
    # Extract the generated text from results
    continuation = results[0] if results else ""
    
    # Build prediction and reference structures for SQuAD metric
    predictions = {
        "id": doc["id"],
        "prediction_text": continuation.strip(),
        "no_answer_probability": 0.0  # Simplified since we're focusing on generation
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