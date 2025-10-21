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

        # For IterableColumn or similar objects, convert to dict
        # by extracting all items
        try:
            # IterableColumn wraps a dict-like structure
            # Try accessing it as a mapping
            if hasattr(doc, '__iter__') and not isinstance(doc, (str, bytes)):
                # This might be an iterable that needs conversion
                # Try converting to list first, then to dict if it's key-value pairs
                items = list(doc)
                if items and isinstance(items[0], tuple) and len(items[0]) == 2:
                    # List of (key, value) tuples
                    return dict(items)
                elif len(items) == 1 and isinstance(items[0], dict):
                    # Single dict item
                    return items[0]
        except Exception as e:
            import sys
            print(f"DEBUG: IterableColumn conversion failed: {e}", file=sys.stderr)

        # Try .to_dict() method
        if hasattr(doc, 'to_dict'):
            try:
                return doc.to_dict()
            except Exception as e:
                import sys
                print(f"DEBUG: .to_dict() failed: {e}", file=sys.stderr)

        # Try manual key extraction for dict-like objects
        if hasattr(doc, 'keys'):
            try:
                keys = list(doc.keys()) if callable(doc.keys) else list(doc.keys)
                return {key: doc[key] for key in keys}
            except Exception as e:
                import sys
                print(f"DEBUG: Manual key extraction failed: {e}, type={type(doc)}", file=sys.stderr)

        # If all conversions fail, return as-is and let downstream handle it
        import sys
        print(f"DEBUG: Returning doc as-is, type={type(doc)}", file=sys.stderr)
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