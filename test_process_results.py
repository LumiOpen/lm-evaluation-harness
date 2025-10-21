#!/usr/bin/env python
"""Test helmet_process_results with answer field."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lm_eval.tasks.helmet.utils import helmet_process_results

print("Testing helmet_process_results...")

# Test cases
test_cases = [
    {
        'name': 'Exact match',
        'doc': {'answer': 'hello world', 'question': 'test'},
        'results': ['hello world'],
        'expected': {'exact_match': 1.0, 'f1': 1.0}
    },
    {
        'name': 'No match',
        'doc': {'answer': 'hello world', 'question': 'test'},
        'results': ['goodbye world'],
        'expected': {'exact_match': 0.0}  # f1 will be calculated
    },
    {
        'name': 'Partial match',
        'doc': {'answer': 'hello world', 'question': 'test'},
        'results': ['hello there'],
        'expected': {'exact_match': 0.0}  # f1 will be calculated
    },
    {
        'name': 'Reranking answer',
        'doc': {'answer': '123 > 456 > 789', 'question': 'rank these'},
        'results': ['123 > 456 > 789'],
        'expected': {'exact_match': 1.0, 'f1': 1.0}
    },
]

all_passed = True

for test in test_cases:
    print(f"\nTest: {test['name']}")
    print(f"  Doc answer: {test['doc']['answer']}")
    print(f"  Results: {test['results']}")

    try:
        result = helmet_process_results(test['doc'], test['results'])
        print(f"  Got: exact_match={result['exact_match']}, f1={result['f1']:.3f}")

        # Check exact_match
        if result['exact_match'] != test['expected']['exact_match']:
            print(f"  ✗ FAIL: Expected exact_match={test['expected']['exact_match']}")
            all_passed = False
        else:
            print(f"  ✓ PASS")

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

if all_passed:
    print("\n" + "="*60)
    print("SUCCESS! All tests passed")
    print("="*60)
else:
    print("\n" + "="*60)
    print("FAILURE! Some tests failed")
    print("="*60)
    sys.exit(1)
