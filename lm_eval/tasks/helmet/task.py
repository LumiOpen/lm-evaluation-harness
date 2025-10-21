"""Custom HELMET task class to handle streaming datasets without core modifications."""

import os
from lm_eval.api.task import ConfigurableTask


class HELMETTask(ConfigurableTask):
    """Custom task class for HELMET that handles streaming datasets safely."""

    def __init__(self, config=None, **kwargs):
        # Filter out custom fields that TaskConfig doesn't understand
        if config:
            filtered_config = {k: v for k, v in config.items() if k != 'class'}
        else:
            filtered_config = config
        super().__init__(config=filtered_config, **kwargs)

    def _process_doc(self, doc):
        """Override to handle streaming dataset documents safely."""
        # Convert dataset row to plain dict if needed (for streaming datasets)
        if hasattr(doc, 'to_dict'):
            return doc.to_dict()
        elif not isinstance(doc, dict):
            # Try to convert to dict (works for most dataset row types)
            return dict(doc)
        return doc

    def doc_iterator(self, rank=0, num_workers=1, limit=None):
        """Override doc_iterator to support HELMET_LIMIT environment variable."""
        # Check if HELMET_LIMIT is set
        helmet_limit = os.environ.get('HELMET_LIMIT')
        if helmet_limit:
            try:
                helmet_limit = int(helmet_limit)
                print(f"HELMET: Limiting dataset to {helmet_limit} samples")
                # Use the smaller of the two limits
                if limit is None or helmet_limit < limit:
                    limit = helmet_limit
            except (ValueError, TypeError):
                print(f"HELMET: Invalid limit '{helmet_limit}', ignoring")

        # Call parent doc_iterator with potentially modified limit
        count = 0
        for doc in super().doc_iterator(rank=rank, num_workers=num_workers, limit=limit):
            if helmet_limit and count >= helmet_limit:
                print(f"HELMET: Reached limit {helmet_limit}, stopping dataset iteration")
                break
            yield doc
            count += 1

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def fewshot_docs(self):
        """Return empty iterator for fewshot - HELMET doesn't use fewshot examples."""
        return iter([])