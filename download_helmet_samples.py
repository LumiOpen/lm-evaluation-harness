#!/usr/bin/env python
"""Download actual HELMET samples and save them locally for inspection."""

import sys
import json
from datasets import load_dataset

print("Downloading HELMET dataset samples...")

# Load in streaming mode
dataset = load_dataset(
    "princeton-nlp/HELMET",
    name="default",
    split="train",
    streaming=True
)

print("Fetching first 10 samples and saving to file...\n")

samples = []
for i, sample in enumerate(dataset):
    if i >= 10:
        break

    print(f"Sample {i+1}:")
    print(f"  Keys: {list(sample.keys())}")

    # Decode jsonl field
    if 'jsonl' in sample:
        jsonl_data = sample['jsonl']
        if isinstance(jsonl_data, bytes):
            jsonl_data = jsonl_data.decode('utf-8')

        # Parse first line
        first_line = jsonl_data.split('\n')[0].strip()
        try:
            parsed = json.loads(first_line)
            print(f"  Parsed keys: {list(parsed.keys())}")

            # Save full sample
            samples.append({
                'sample_num': i+1,
                'raw_keys': list(sample.keys()),
                'parsed_keys': list(parsed.keys()),
                'parsed_data': parsed,
                '__key__': sample.get('__key__', ''),
            })
        except Exception as e:
            print(f"  ERROR parsing: {e}")

# Save to file
output_file = '/tmp/helmet_samples.json'
with open(output_file, 'w') as f:
    json.dump(samples, f, indent=2)

print(f"\n✓ Saved {len(samples)} samples to {output_file}")

# Analyze field patterns
print("\n" + "="*60)
print("FIELD ANALYSIS")
print("="*60)

all_fields = set()
for s in samples:
    all_fields.update(s['parsed_keys'])

print(f"\nAll unique fields found: {sorted(all_fields)}")

# Group by field combinations
field_combos = {}
for s in samples:
    combo = tuple(sorted(s['parsed_keys']))
    if combo not in field_combos:
        field_combos[combo] = []
    field_combos[combo].append(s['sample_num'])

print(f"\nField combinations:")
for combo, sample_nums in field_combos.items():
    print(f"  {combo}")
    print(f"    Found in samples: {sample_nums}")

print(f"\n✓ Inspect the full data in: {output_file}")
