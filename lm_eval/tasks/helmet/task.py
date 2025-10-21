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
        """Convert dataset row to plain dict if needed (for streaming datasets)."""
        if isinstance(doc, dict):
            return doc
        if hasattr(doc, 'to_dict'):
            return doc.to_dict()
        # Try to convert via dict() constructor (works for dict-like objects)
        try:
            return dict(doc)
        except (TypeError, ValueError):
            # Last resort: if it has column-like access, extract manually
            if hasattr(doc, 'keys') and hasattr(doc, '__getitem__'):
                try:
                    return {key: doc[key] for key in doc.keys()}
                except Exception:
                    pass
        # If all conversions fail, return as-is and let downstream handle it
        return doc

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def fewshot_docs(self):
        """Return empty iterator for fewshot - HELMET doesn't use fewshot examples."""
        return iter([])

    def doc_to_text(self, doc):
        """Convert doc to dict before processing."""
        doc = self._process_doc(doc)
        return super().doc_to_text(doc)

    def doc_to_target(self, doc):
        """Convert doc to dict before processing."""
        doc = self._process_doc(doc)
        return super().doc_to_target(doc)