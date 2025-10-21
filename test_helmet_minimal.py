#!/usr/bin/env python
"""Minimal test for HELMET dataset parsing - downloads only 1 sample."""

import sys
sys.path.insert(0, '/Users/danielzautner/lm-evaluation-harness')

from lm_eval.tasks import TaskManager

print("Initializing task manager...")
task_manager = TaskManager()

print("Loading helmet_recall task...")
try:
    task_dict = task_manager.load_task_or_group("helmet_recall")
    print(f"✓ Task loaded successfully: {list(task_dict.keys())}")

    # Get the task
    task = task_dict['helmet_recall']
    print(f"✓ Task type: {type(task)}")

    # Try to get test docs
    print("\nGetting test docs (will download ~1 sample)...")
    test_docs = task.test_docs()

    # Get first doc
    print("Getting first document...")
    first_doc = test_docs[0]
    print(f"✓ First doc type: {type(first_doc)}")
    print(f"✓ First doc keys: {list(first_doc.keys())[:10]}")

    # Try doc_to_text
    print("\nTesting doc_to_text...")
    text = task.doc_to_text(first_doc)
    print(f"✓ doc_to_text result (first 200 chars): {text[:200]}")

    # Try doc_to_target
    print("\nTesting doc_to_target...")
    target = task.doc_to_target(first_doc)
    print(f"✓ doc_to_target result (first 200 chars): {str(target)[:200]}")

    print("\n" + "="*60)
    print("SUCCESS! HELMET task is working correctly.")
    print("="*60)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
