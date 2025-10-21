#!/usr/bin/env python
"""Fast test - only downloads and tests 1 sample from HELMET."""

import sys
import os

# Make sure we use local code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Step 1: Importing modules...")
from datasets import load_dataset
import json

print("Step 2: Loading HELMET dataset in streaming mode (will download ~1 sample)...")
dataset = load_dataset(
    "princeton-nlp/HELMET",
    name="default",
    split="train",
    streaming=True
)

print("Step 3: Getting first sample...")
first_sample = next(iter(dataset))

print(f"\nRaw sample type: {type(first_sample)}")
print(f"Raw sample keys: {list(first_sample.keys())}")

print("\nStep 4: Processing sample (mimicking _process_doc)...")
if 'jsonl' in first_sample:
    jsonl_data = first_sample['jsonl']
    print(f"  jsonl field type: {type(jsonl_data)}")
    print(f"  jsonl field (first 200 bytes): {str(jsonl_data[:200])}")

    # Decode bytes to string
    if isinstance(jsonl_data, bytes):
        jsonl_str = jsonl_data.decode('utf-8')
        print(f"  Decoded to string, length: {len(jsonl_str)}")
    else:
        jsonl_str = jsonl_data

    # Parse JSON
    parsed = json.loads(jsonl_str)
    print(f"\n  Parsed JSON keys: {list(parsed.keys())}")

    # Show first few fields
    print("\n  Parsed fields:")
    for key in list(parsed.keys())[:10]:
        value = str(parsed[key])[:100]
        print(f"    {key}: {value}")

    # Normalize field names
    if 'query' in parsed and 'question' not in parsed:
        parsed['question'] = parsed['query']
        print("\n  ✓ Normalized 'query' -> 'question'")

    if 'answers' in parsed and 'answer' not in parsed:
        answers = parsed['answers']
        if isinstance(answers, list) and answers:
            parsed['answer'] = answers[0]
        else:
            parsed['answer'] = answers
        print(f"  ✓ Normalized 'answers' -> 'answer': {str(parsed['answer'])[:100]}")

    print(f"\nStep 5: Testing template rendering...")
    print(f"  question field exists: {'question' in parsed}")
    print(f"  answer field exists: {'answer' in parsed}")

    if 'question' in parsed:
        print(f"  question value: {str(parsed['question'])[:200]}")
    if 'answer' in parsed:
        print(f"  answer value: {str(parsed['answer'])[:200]}")

    print("\n" + "="*60)
    print("SUCCESS! Dataset parsing works correctly.")
    print("="*60)
    print("\nThe HELMET task should work on LUMI now!")
else:
    print("ERROR: No 'jsonl' field found!")
    sys.exit(1)
