import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        label_map = {
            "correct": "kyllä",
            "incorrect": "ei"
        }

        out_doc = {
            "query": f"Lause: {doc['text']}\nKieliopillisesti oikein: ",
            "choices": [label_map['correct'], label_map['incorrect']],
            "gold": label_map[doc['label']]
        }
        return out_doc

    return dataset.map(_process_doc)