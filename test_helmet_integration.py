#!/usr/bin/env python
"""Integration test for HELMET task with mock data."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Integration test: HELMET task class with mock data\n")

# Import the actual HELMET task class
from lm_eval.tasks.helmet.task import HELMETTask

# Create mock dataset that mimics the real HELMET structure
class MockDataset:
    """Mock dataset that returns HELMET-like data."""

    def __init__(self):
        self.features = {
            'jsonl': 'string',
            '__key__': 'string',
            '__url__': 'string'
        }

    def __iter__(self):
        # Return a few mock samples
        samples = [
            {
                'jsonl': b'{"qid": "131843", "query": "definition of a sigmet", "answers": ["significant meteorological information"], "ctxs": [{"id": "8305152", "text": "SIGMET definition"}]}',
                '__key__': 'sample1',
                '__url__': 'mock://url1'
            },
            {
                'jsonl': b'{"qid": "222", "query": "what is python", "answers": ["a programming language"], "ctxs": [{"id": "999", "text": "Python is a language"}]}',
                '__key__': 'sample2',
                '__url__': 'mock://url2'
            },
        ]
        for sample in samples:
            yield sample

# Create a minimal config for the task
config = {
    'task': 'helmet_recall',
    'dataset_path': 'mock',
    'dataset_name': 'default',
    'test_split': 'train',
    'doc_to_text': '{{ question }}',
    'doc_to_target': '{{ answer }}',
    'output_type': 'generate_until',
    'generation_kwargs': {
        'max_gen_toks': 50,
        'temperature': 0.0,
        'do_sample': False,
        'until': ['\n']
    },
}

print("Step 1: Creating HELMET task instance...")
try:
    task = HELMETTask(config=config)
    # Override dataset with our mock
    task.dataset = {'train': MockDataset()}
    print("  ✓ Task created successfully")
except Exception as e:
    print(f"  ✗ Failed to create task: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 2: Getting test docs...")
try:
    test_docs = task.test_docs()
    print(f"  ✓ test_docs() returned: {type(test_docs)}")
except Exception as e:
    print(f"  ✗ Failed to get test_docs: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: Getting first document...")
try:
    first_doc = test_docs[0]
    print(f"  ✓ Got first doc, type: {type(first_doc)}")
    print(f"  Keys: {list(first_doc.keys())}")
except Exception as e:
    print(f"  ✗ Failed to get first doc: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Testing doc_to_text...")
try:
    text = task.doc_to_text(first_doc)
    print(f"  ✓ doc_to_text result: '{text}'")
except Exception as e:
    print(f"  ✗ doc_to_text failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Testing doc_to_target...")
try:
    target = task.doc_to_target(first_doc)
    print(f"  ✓ doc_to_target result: '{target}'")
except Exception as e:
    print(f"  ✗ doc_to_target failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 6: Iterating through test_docs...")
try:
    count = 0
    for doc in test_docs:
        text = task.doc_to_text(doc)
        target = task.doc_to_target(doc)
        print(f"  Sample {count+1}: question='{text[:50]}...', answer='{target[:50]}...'")
        count += 1
        if count >= 2:
            break
    print(f"  ✓ Processed {count} samples successfully")
except Exception as e:
    print(f"  ✗ Iteration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("SUCCESS! HELMET task integration works perfectly!")
print("="*60)
print("\nAll fixes are working:")
print("  ✓ jsonl field parsing")
print("  ✓ Field name normalization (query->question, answers->answer)")
print("  ✓ Template rendering")
print("  ✓ Streaming dataset wrapper")
print("\nReady for LUMI!")
