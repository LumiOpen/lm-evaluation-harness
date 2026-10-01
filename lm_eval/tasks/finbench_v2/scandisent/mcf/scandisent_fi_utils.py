import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        label_map = {
            "positive": "positiivinen",
            "negative": "negatiivinen"
        }

        out_doc = {
            "query": doc['text'],
            "choices": [label_map['positive'], label_map['negative']],
            "gold": label_map[doc['label']]
        }
        return out_doc

    return dataset.map(_process_doc)