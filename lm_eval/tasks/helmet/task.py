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

        # Try .to_dict() method first
        if hasattr(doc, 'to_dict'):
            try:
                return doc.to_dict()
            except Exception:
                pass

        # Try manual key extraction for dict-like objects
        if hasattr(doc, 'keys') and hasattr(doc, '__getitem__'):
            try:
                return {key: doc[key] for key in doc.keys()}
            except Exception:
                pass

        # Try dict() constructor as fallback
        try:
            return dict(doc)
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