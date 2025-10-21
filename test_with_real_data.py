#!/usr/bin/env python
"""Test HELMET task processing with REAL data from HuggingFace."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("TESTING WITH REAL HELMET DATA")
print("="*60)

print("\nStep 1: Downloading first REAL sample from HELMET dataset...")
from datasets import load_dataset

dataset = load_dataset(
    "princeton-nlp/HELMET",
    name="default",
    split="train",
    streaming=True
)

real_sample = next(iter(dataset))
print(f"  ✓ Downloaded sample")
print(f"  Raw keys: {list(real_sample.keys())}")

print("\nStep 2: Processing with HELMETTask._process_doc...")
from lm_eval.tasks.helmet.task import HELMETTask

# Create minimal task instance
class TestHELMETTask(HELMETTask):
    def __init__(self):
        self._cached_first_doc = None

task = TestHELMETTask()

# Process the real sample
processed = task._process_doc(real_sample)

print(f"\n  ✓ Processed successfully")
print(f"  Processed keys: {list(processed.keys())}")

# Check critical fields
print("\nStep 3: Verifying required fields...")
required_fields = ['question', 'answer']
for field in required_fields:
    if field in processed:
        value = str(processed[field])[:100]
        print(f"  ✓ {field}: {value}")
    else:
        print(f"  ✗ MISSING: {field}")
        sys.exit(1)

print("\nStep 4: Testing Jinja2 template rendering...")
from jinja2 import Template

doc_to_text = Template("{{ question }}")
doc_to_target = Template("{{ answer }}")

try:
    text = doc_to_text.render(**processed)
    target = doc_to_target.render(**processed)
    print(f"  ✓ question rendered: {text[:100]}")
    print(f"  ✓ answer rendered: {target[:100]}")
except Exception as e:
    print(f"  ✗ Template rendering failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("SUCCESS! REAL DATA TEST PASSED")
print("="*60)
print("\nThe fix works with actual HELMET data from HuggingFace!")
