#!/usr/bin/env python
"""Quick local test for HELMET tasks without needing a real model."""

import sys
from datasets import load_dataset

# Test loading the HELMET dataset in streaming mode
print("Loading HELMET dataset in streaming mode...")
dataset = load_dataset(
    "princeton-nlp/HELMET",
    name="default",
    split="train",
    streaming=True
)

print(f"\nDataset type: {type(dataset)}")
print(f"Dataset features: {dataset.features if hasattr(dataset, 'features') else 'N/A'}")

# Get first item
print("\nGetting first item from dataset...")
first_item = next(iter(dataset))
print(f"First item type: {type(first_item)}")
print(f"First item keys: {list(first_item.keys()) if hasattr(first_item, 'keys') else 'N/A'}")

# Print first few items to see structure
print("\n=== First 3 items from dataset ===")
dataset = load_dataset(
    "princeton-nlp/HELMET",
    name="default",
    split="train",
    streaming=True
)

for i, item in enumerate(dataset):
    if i >= 3:
        break
    print(f"\nItem {i}:")
    print(f"  Type: {type(item)}")
    if isinstance(item, dict):
        print(f"  Keys: {list(item.keys())}")
        for key in list(item.keys())[:5]:  # Show first 5 keys
            value = item[key]
            value_str = str(value)[:100] if value else str(value)
            print(f"    {key}: {value_str}")
    else:
        print(f"  Value: {str(item)[:200]}")
