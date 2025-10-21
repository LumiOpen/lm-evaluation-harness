#!/usr/bin/env python
"""Test that StreamingDatasetWrapper has __len__."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the wrapper directly
print("Creating mock StreamingDatasetWrapper...")

class MockTask:
    def __init__(self):
        self._cached_first_doc = None

    def _process_doc(self, doc):
        return doc

class MockDataset:
    def __init__(self):
        self.features = {}

    def __iter__(self):
        yield {'jsonl': b'{"qid": "1", "query": "test", "answers": ["ans"]}\n', '__key__': 'k1', '__url__': 'u1'}

# Import the wrapper code
exec(open('/Users/danielzautner/lm-evaluation-harness/lm_eval/tasks/helmet/task.py').read().split('class StreamingDatasetWrapper:')[1].split('return StreamingDatasetWrapper')[0])

test_docs = locals()['StreamingDatasetWrapper'](MockDataset(), MockTask())

print(f"Type of test_docs: {type(test_docs)}")

# Test __len__
try:
    length = len(test_docs)
    print(f"✓ len(test_docs) = {length}")
except TypeError as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# Test iteration still works
try:
    first_doc = next(iter(test_docs))
    print(f"✓ Iteration works, got doc with keys: {list(first_doc.keys())}")
except Exception as e:
    print(f"✗ Iteration failed: {e}")
    sys.exit(1)

# Test [0] still works
try:
    doc_0 = test_docs[0]
    print(f"✓ Index access works, got doc with keys: {list(doc_0.keys())}")
except Exception as e:
    print(f"✗ Index access failed: {e}")
    sys.exit(1)

print("\n✓ All tests passed!")
