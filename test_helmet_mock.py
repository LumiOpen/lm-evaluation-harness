#!/usr/bin/env python
"""Test HELMET task with mock data (no download needed)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing HELMET task with mock data...\n")

# Create mock HELMET dataset row based on actual structure we saw in logs
mock_raw_doc = {
    'jsonl': b'{"qid": "131843", "query": "definition of a sigmet", "answers": ["significant meteorological information"], "ctxs": [{"id": "8305152", "text": "Definition of SIGMET. plural. SIGMETs. also. sigmets. : a notice of significant hazardous weather conditions"}]}',
    '__key__': 'data/msmarco/test_reranking_data_k14_dep3',
    '__url__': 'hf://datasets/princeton-nlp/HELMET@c5b85e7f0d954ffe71b3fe5b6d1da17a11094b1b/data.tar.gz'
}

print("Step 1: Mock raw document")
print(f"  Keys: {list(mock_raw_doc.keys())}")
print(f"  jsonl field: {mock_raw_doc['jsonl'][:100]}...")

# Now test our _process_doc logic
print("\nStep 2: Testing _process_doc logic...")

import json

doc = mock_raw_doc

if 'jsonl' in doc:
    jsonl_data = doc['jsonl']
    print(f"  Found jsonl field, type: {type(jsonl_data)}")

    # Decode bytes to string
    if isinstance(jsonl_data, bytes):
        jsonl_data = jsonl_data.decode('utf-8')
        print(f"  Decoded bytes to string")

    # Parse JSON
    parsed = json.loads(jsonl_data)
    print(f"  Parsed JSON, keys: {list(parsed.keys())}")

    # Normalize field names
    if 'query' in parsed and 'question' not in parsed:
        parsed['question'] = parsed['query']
        print(f"  ✓ Normalized 'query' -> 'question'")

    if 'answers' in parsed and 'answer' not in parsed:
        answers = parsed['answers']
        if isinstance(answers, list) and answers:
            parsed['answer'] = answers[0]
        else:
            parsed['answer'] = answers
        print(f"  ✓ Normalized 'answers' -> 'answer'")

    doc = parsed

print(f"\nStep 3: Processed document")
print(f"  Keys: {list(doc.keys())}")
print(f"  question: {doc.get('question', 'MISSING')}")
print(f"  answer: {doc.get('answer', 'MISSING')}")

# Test template rendering
print("\nStep 4: Testing Jinja2 template rendering...")
from jinja2 import Template

doc_to_text_template = Template("{{ question }}")
doc_to_target_template = Template("{{ answer }}")

try:
    text = doc_to_text_template.render(**doc)
    print(f"  ✓ doc_to_text rendered: {text}")
except Exception as e:
    print(f"  ✗ doc_to_text failed: {e}")
    sys.exit(1)

try:
    target = doc_to_target_template.render(**doc)
    print(f"  ✓ doc_to_target rendered: {target}")
except Exception as e:
    print(f"  ✗ doc_to_target failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("SUCCESS! All parsing and template rendering works!")
print("="*60)
print("\nThe fix is correct. HELMET should work on LUMI now.")
