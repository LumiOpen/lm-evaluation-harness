#!/usr/bin/env python
"""Download and inspect actual HELMET dataset structure."""

import sys
from datasets import load_dataset

print("Downloading first sample from HELMET dataset...")
print("(This will download a small portion of the dataset)\n")

# Load in streaming mode to avoid downloading everything
dataset = load_dataset(
    "princeton-nlp/HELMET",
    name="default",
    split="train",
    streaming=True
)

print("Getting first sample...")
first_sample = next(iter(dataset))

print("\n" + "="*60)
print("RAW SAMPLE STRUCTURE")
print("="*60)

print(f"\nSample type: {type(first_sample)}")
print(f"Sample keys: {list(first_sample.keys())}")

for key in first_sample.keys():
    value = first_sample[key]
    print(f"\n--- Key: {key} ---")
    print(f"Type: {type(value)}")

    if isinstance(value, bytes):
        print(f"Length: {len(value)} bytes")
        print(f"First 500 bytes:\n{value[:500]}")
        print(f"\nLast 500 bytes:\n{value[-500:]}")

        # Decode and show structure
        try:
            decoded = value.decode('utf-8')
            print(f"\nDecoded length: {len(decoded)} characters")

            # Check for newlines
            lines = decoded.split('\n')
            print(f"Number of lines (by \\n): {len(lines)}")

            if len(lines) > 1:
                print(f"\nFirst line ({len(lines[0])} chars):\n{lines[0][:200]}")
                print(f"\nSecond line ({len(lines[1])} chars):\n{lines[1][:200]}")
                if len(lines) > 2:
                    print(f"\nThird line ({len(lines[2])} chars):\n{lines[2][:200]}")

            # Try to parse as JSON
            import json
            print("\nAttempting to parse entire field as single JSON...")
            try:
                parsed = json.loads(decoded)
                print(f"✓ SUCCESS - it's a single JSON object")
                print(f"  Keys: {list(parsed.keys())[:10]}")
            except json.JSONDecodeError as e:
                print(f"✗ FAILED - {e}")
                print("\nAttempting to parse first line only...")
                try:
                    first_line = lines[0].strip()
                    parsed = json.loads(first_line)
                    print(f"✓ SUCCESS - first line is valid JSON")
                    print(f"  Keys: {list(parsed.keys())}")
                except json.JSONDecodeError as e2:
                    print(f"✗ FAILED - {e2}")
        except Exception as e:
            print(f"Error decoding: {e}")
    else:
        print(f"Value: {str(value)[:200]}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
