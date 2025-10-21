#!/bin/bash
# Local test script for HELMET tasks using dummy backend

set -x  # Print commands as they run

python -m lm_eval \
  --model dummy \
  --model_args pretrained=dummy \
  --tasks helmet_recall \
  --limit 1 \
  --output_path /tmp/helmet_test_output \
  --log_samples

echo "Test complete. Check output above for DEBUG messages showing dataset structure."
