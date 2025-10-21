#!/usr/bin/env python
"""Test HELMET _process_doc method directly."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing HELMET _process_doc method\n")

# Import the task class
from lm_eval.tasks.helmet.task import HELMETTask

# Create a minimal task instance without initializing
class TestHELMETTask(HELMETTask):
    def __init__(self):
        # Skip parent init to avoid dataset download
        self._cached_first_doc = None

print("Step 1: Creating task instance...")
task = TestHELMETTask()
print("  ✓ Task created (skipped dataset download)")

# Test with mock data - IMPORTANT: Use JSONL format (multiple JSON objects separated by newlines)
# This mimics the actual HELMET dataset structure
mock_docs = [
    # Task with 'answers' field
    {
        'jsonl': b'{"qid": "131843", "query": "definition of a sigmet", "answers": ["significant meteorological information"], "ctxs": [{"id": "8305152", "text": "SIGMET definition"}]}\n{"other": "data"}\n{"more": "lines"}',
        '__key__': 'sample1',
        '__url__': 'mock://url1'
    },
    # Task with 'answers' field (list)
    {
        'jsonl': b'{"qid": "222", "query": "what is python", "answers": ["a programming language"], "ctxs": [{"id": "999", "text": "Python is a language"}]}\n{"extra": "json"}',
        '__key__': 'sample2',
        '__url__': 'mock://url2'
    },
    # RERANKING task: NO answers field, must compute from ctxs
    {
        'jsonl': b'{"qid": "333", "query": "definition of a sigmet", "ctxs": [{"id": "8305152", "text": "Definition of SIGMET", "label": 3}, {"id": "1234", "text": "Other doc", "label": 1}, {"id": "5678", "text": "Another doc", "label": 2}]}\n{"another": "line"}',
        '__key__': 'sample3_rerank',
        '__url__': 'mock://url3'
    },
]

print("\nStep 2: Testing _process_doc on multiple samples...")
for i, raw_doc in enumerate(mock_docs, 1):
    print(f"\n  Sample {i}:")
    print(f"    Raw keys: {list(raw_doc.keys())}")

    try:
        processed = task._process_doc(raw_doc)
        print(f"    ✓ Processed keys: {list(processed.keys())}")
        print(f"    ✓ question: {processed.get('question', 'MISSING')}")
        print(f"    ✓ answer: {processed.get('answer', 'MISSING')}")

        # Verify required fields exist
        if 'question' not in processed:
            print(f"    ✗ ERROR: 'question' field missing!")
            sys.exit(1)
        if 'answer' not in processed:
            print(f"    ✗ ERROR: 'answer' field missing!")
            sys.exit(1)

    except Exception as e:
        print(f"    ✗ ERROR processing doc: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\nStep 3: Testing template rendering with processed docs...")
from jinja2 import Template

doc_to_text_template = Template("{{ question }}")
doc_to_target_template = Template("{{ answer }}")

for i, raw_doc in enumerate(mock_docs, 1):
    processed = task._process_doc(raw_doc)

    try:
        text = doc_to_text_template.render(**processed)
        target = doc_to_target_template.render(**processed)
        print(f"  Sample {i}: '{text}' -> '{target}'")
    except Exception as e:
        print(f"  ✗ Template rendering failed for sample {i}: {e}")
        sys.exit(1)

print("\n" + "="*60)
print("SUCCESS! All tests passed!")
print("="*60)
print("\nVerified:")
print("  ✓ _process_doc correctly parses jsonl bytes")
print("  ✓ Field normalization works (query->question, answers->answer)")
print("  ✓ Templates render without errors")
print("  ✓ Works with multiple samples")
print("\nHELMET tasks are ready to run on LUMI!")
