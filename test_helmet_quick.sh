#!/bin/bash
# Quick local test for HELMET - only processes 1 sample to avoid downloading 6.89 GB

set -ex

# Install minimal dependencies if needed
python -m pip install -q lm_eval 2>/dev/null || true

# Test with dummy model (no actual inference) and limit to 1 sample
python -m lm_eval \
  --model dummy \
  --model_args pretrained=dummy \
  --tasks helmet_recall \
  --limit 1 \
  --output_path /tmp/helmet_quick_test \
  --log_samples 2>&1 | tee /tmp/helmet_test_output.log

echo ""
echo "==== Test Summary ===="
if grep -q "question" /tmp/helmet_test_output.log; then
    echo "✓ Dataset fields parsed correctly"
else
    echo "✗ Dataset parsing may have issues"
fi

if grep -q "UndefinedError" /tmp/helmet_test_output.log; then
    echo "✗ Template error found"
    exit 1
else
    echo "✓ No template errors"
fi

echo ""
echo "Full log saved to: /tmp/helmet_test_output.log"
