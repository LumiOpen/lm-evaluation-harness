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

        # Cache first doc for testing during init (streaming datasets can't be indexed)
        self._cached_first_doc = None

        super().__init__(config=filtered_config, **kwargs)

    def _process_doc(self, doc):
        """Convert dataset row to plain dict if needed (for streaming datasets)."""
        import sys
        doc_type = str(type(doc))
        print(f"DEBUG _process_doc called: type={doc_type}", file=sys.stderr)

        # Check if it's actually IterableColumn masquerading as a dict
        if 'IterableColumn' in doc_type:
            print(f"DEBUG: Detected IterableColumn, attempting conversion", file=sys.stderr)
            # Force convert IterableColumn by iterating and extracting values
            try:
                if hasattr(doc, '__iter__'):
                    items = list(doc)
                    print(f"DEBUG: Converted to list with {len(items)} items", file=sys.stderr)
                    if items and isinstance(items[0], dict):
                        print(f"DEBUG: Returning first dict item", file=sys.stderr)
                        return items[0]
                    print(f"DEBUG: Items type: {type(items[0]) if items else 'empty'}", file=sys.stderr)
            except Exception as e:
                print(f"DEBUG: IterableColumn iteration failed: {e}", file=sys.stderr)

        if isinstance(doc, dict) and 'IterableColumn' not in doc_type:
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
            print(f"DEBUG: IterableColumn list conversion failed: {e}", file=sys.stderr)

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

    def test_docs(self):
        """Override to provide iterable-compatible access to test docs."""
        # Wrap the dataset in a helper that caches the first doc for indexed access
        dataset = self.dataset[self.config.test_split]
        if self.config.process_docs is not None:
            dataset = self.config.process_docs(dataset)

        # Return a wrapper that supports both iteration and [0] access
        class StreamingDatasetWrapper:
            def __init__(self, dataset, task):
                self.dataset = dataset
                self.task = task

            def __iter__(self):
                for doc in self.dataset:
                    yield doc

            def __getitem__(self, idx):
                if idx == 0:
                    # Cache and return first doc
                    if self.task._cached_first_doc is None:
                        # Get first item - might be IterableColumn or dict
                        first_item = next(iter(self.dataset))

                        # If it's IterableColumn, we need to extract the actual data
                        # IterableColumn wraps column data - we need to get the underlying dict
                        import sys
                        print(f"DEBUG __getitem__: first_item type = {type(first_item)}", file=sys.stderr)

                        if hasattr(first_item, 'to_dict'):
                            first_item = first_item.to_dict()
                        elif not isinstance(first_item, dict):
                            # Try to extract dict from IterableColumn or similar
                            print(f"DEBUG: first_item has these attrs: {dir(first_item)[:10]}", file=sys.stderr)

                        self.task._cached_first_doc = first_item
                    return self.task._cached_first_doc
                raise IndexError("Only index 0 is supported for streaming datasets")

            @property
            def features(self):
                return self.dataset.features if hasattr(self.dataset, 'features') else {}

        return StreamingDatasetWrapper(dataset, self)

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