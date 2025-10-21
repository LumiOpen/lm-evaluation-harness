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
        import json
        doc_type = str(type(doc))
        print(f"DEBUG _process_doc called: type={doc_type}", file=sys.stderr)

        # If it's already a plain dict, check if we need to parse jsonl field
        if isinstance(doc, dict) and 'IterableColumn' not in doc_type:
            # HELMET dataset stores actual data in a 'jsonl' field as bytes
            if 'jsonl' in doc:
                try:
                    jsonl_data = doc['jsonl']
                    # Decode bytes to string if needed
                    if isinstance(jsonl_data, bytes):
                        jsonl_data = jsonl_data.decode('utf-8')

                    # JSONL format = multiple JSON objects separated by newlines
                    # Parse only the first line
                    first_line = jsonl_data.split('\n')[0].strip()
                    if not first_line:
                        print(f"DEBUG _process_doc: Empty jsonl field", file=sys.stderr)
                        return doc

                    # Parse first JSON object
                    parsed = json.loads(first_line)
                    print(f"DEBUG _process_doc: Parsed jsonl field, got keys: {list(parsed.keys())}", file=sys.stderr)

                    # Normalize field names - HELMET uses different names across tasks
                    # Map common variations to standard names
                    if 'query' in parsed and 'question' not in parsed:
                        parsed['question'] = parsed['query']
                    if 'answers' in parsed and 'answer' not in parsed:
                        # Handle both list and single answer
                        answers = parsed['answers']
                        if isinstance(answers, list) and answers:
                            parsed['answer'] = answers[0]
                        else:
                            parsed['answer'] = answers

                    return parsed
                except Exception as e:
                    print(f"DEBUG _process_doc: Failed to parse jsonl field: {e}", file=sys.stderr)
            return doc

        # Try accessing internal batch data structures (common in HF datasets)
        for attr in ['_data', '_batch', 'data', 'batch']:
            if hasattr(doc, attr):
                try:
                    internal_data = getattr(doc, attr)
                    print(f"DEBUG: Found {attr} attribute: {type(internal_data)}", file=sys.stderr)
                    if isinstance(internal_data, dict):
                        return internal_data
                    elif isinstance(internal_data, list) and internal_data and isinstance(internal_data[0], dict):
                        return internal_data[0]
                except Exception as e:
                    print(f"DEBUG: Accessing {attr} failed: {e}", file=sys.stderr)

        # Try .to_dict() method
        if hasattr(doc, 'to_dict'):
            try:
                result = doc.to_dict()
                print(f"DEBUG: .to_dict() returned type={type(result)}", file=sys.stderr)
                if isinstance(result, dict):
                    return result
            except Exception as e:
                print(f"DEBUG: .to_dict() failed: {e}", file=sys.stderr)

        # Try manual key extraction for dict-like objects
        if hasattr(doc, 'keys') and hasattr(doc, '__getitem__'):
            try:
                keys_obj = doc.keys()
                keys = list(keys_obj) if not isinstance(keys_obj, list) else keys_obj
                print(f"DEBUG: Found keys: {keys[:5] if len(keys) > 5 else keys}", file=sys.stderr)
                result = {}
                for key in keys:
                    try:
                        result[key] = doc[key]
                    except Exception as e:
                        print(f"DEBUG: Failed to get doc[{key}]: {e}", file=sys.stderr)
                if result:
                    print(f"DEBUG: Manual key extraction succeeded with {len(result)} keys", file=sys.stderr)
                    return result
            except Exception as e:
                print(f"DEBUG: Manual key extraction failed: {e}", file=sys.stderr)

        # Try to get attributes as dict keys
        if hasattr(doc, '__dict__'):
            try:
                result = {k: v for k, v in doc.__dict__.items() if not k.startswith('_')}
                if result:
                    print(f"DEBUG: Extracted __dict__ with {len(result)} keys", file=sys.stderr)
                    return result
            except Exception as e:
                print(f"DEBUG: __dict__ extraction failed: {e}", file=sys.stderr)

        # Last resort: try to iterate and see what we get
        if hasattr(doc, '__iter__') and not isinstance(doc, (str, bytes)):
            try:
                # Peek at first few items to understand structure
                iterator = iter(doc)
                first_items = []
                for _ in range(min(5, 100)):  # Safety limit
                    try:
                        item = next(iterator)
                        first_items.append(item)
                    except StopIteration:
                        break

                print(f"DEBUG: Iteration yielded {len(first_items)} items", file=sys.stderr)
                if first_items:
                    print(f"DEBUG: First item type: {type(first_items[0])}", file=sys.stderr)
                    if isinstance(first_items[0], dict):
                        return first_items[0]
                    elif isinstance(first_items[0], tuple) and len(first_items[0]) == 2:
                        return dict(first_items)
            except Exception as e:
                print(f"DEBUG: Iteration failed: {e}", file=sys.stderr)

        # If all conversions fail, return as-is and let downstream handle it
        print(f"DEBUG: All conversion attempts failed, returning as-is: type={type(doc)}", file=sys.stderr)
        print(f"DEBUG: Available attributes: {[a for a in dir(doc) if not a.startswith('_')][:10]}", file=sys.stderr)
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
                    # Process each doc to convert IterableColumn to dict
                    yield self.task._process_doc(doc)

            def __getitem__(self, idx):
                if idx == 0:
                    # Cache and return first doc
                    if self.task._cached_first_doc is None:
                        # Get first item - might be IterableColumn or dict
                        first_item = next(iter(self.dataset))

                        # Always process through _process_doc to handle conversion
                        first_item = self.task._process_doc(first_item)

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
        import sys
        print(f"DEBUG doc_to_text: doc type={type(doc)}, keys={list(doc.keys()) if isinstance(doc, dict) else 'NOT A DICT'}", file=sys.stderr)
        if isinstance(doc, dict):
            print(f"DEBUG doc_to_text: All keys in doc: {list(doc.keys())}", file=sys.stderr)
            # Print first 200 chars of each key's value
            for key in list(doc.keys())[:10]:  # Limit to first 10 keys
                value_str = str(doc[key])[:200]
                print(f"DEBUG doc_to_text:   {key} = {value_str}", file=sys.stderr)
        return super().doc_to_text(doc)

    def doc_to_target(self, doc):
        """Convert doc to dict before processing."""
        doc = self._process_doc(doc)
        return super().doc_to_target(doc)