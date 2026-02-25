"""Tests for crosslingual_longppl process_results and aggregation."""

import math

import pytest

from lm_eval.tasks.crosslingual_longppl.task import (
    longppl_agg,
    nanmean,
    process_results,
)


class FakeRichLoglikelihood(float):
    """Mimics RichLoglikelihood for testing without model dependency."""

    def __new__(cls, value, *, losses=None, offset_mapping=None):
        obj = super().__new__(cls, value)
        obj.losses = losses
        obj.offset_mapping = offset_mapping
        return obj


class TestProcessResults:
    """Test process_results fail-fast and metric computation."""

    def test_missing_losses_raises(self):
        """Missing .losses raises RuntimeError (not NaN)."""
        doc = {"text": "hello world", "key_char_spans": [[0, 5]]}
        loglikelihood = -5.0  # plain float, no .losses
        with pytest.raises(RuntimeError, match="per-token loss data"):
            process_results(doc, [loglikelihood])

    def test_missing_offset_mapping_raises(self):
        """Missing .offset_mapping raises RuntimeError."""
        ll = FakeRichLoglikelihood(-5.0, losses=[1.0, 2.0])
        doc = {"text": "hello world", "key_char_spans": [[0, 5]]}
        with pytest.raises(RuntimeError, match="per-token loss data"):
            process_results(doc, [ll])

    def test_no_global_state_leakage(self):
        """Each call is stateless — second call with plain float still raises."""
        # First call succeeds
        ll_good = FakeRichLoglikelihood(
            -3.0,
            losses=[1.0, 1.5, 0.5],
            offset_mapping=[(0, 3), (3, 6), (6, 9)],
        )
        doc = {"text": "hello wor", "key_char_spans": [[0, 9]]}
        process_results(doc, [ll_good])  # should work

        # Second call with plain float should STILL raise
        ll_bad = -5.0
        with pytest.raises(RuntimeError, match="per-token loss data"):
            process_results(doc, [ll_bad])

    def test_valid_result_returns_metrics(self):
        """Valid RichLoglikelihood produces all expected metric keys."""
        ll = FakeRichLoglikelihood(
            -6.0,
            losses=[1.0, 2.0, 3.0],
            offset_mapping=[(0, 3), (3, 6), (6, 9)],
        )
        doc = {
            "text": "abcdefghi",
            "key_char_spans": [[0, 9]],
        }
        result = process_results(doc, [ll])
        assert "longppl" in result
        assert "mean_key_loss" in result
        assert "key_span_coverage" in result
        assert "word_perplexity" in result
        assert "byte_perplexity" in result
        assert "bits_per_byte" in result

    def test_truncation_denominator_uses_scored_span(self):
        """Perplexity denominators should use scored (truncated) text, not full."""
        full_text = "short " + "x" * 1000  # 1006 chars total
        # offset_mapping covers only first 6 chars (simulating truncation)
        ll = FakeRichLoglikelihood(
            -3.0,
            losses=[1.0, 1.5],
            offset_mapping=[(0, 3), (3, 6)],
        )
        doc = {"text": full_text, "key_char_spans": [[0, 3]]}
        result = process_results(doc, [ll])

        # word_perplexity tuple: (loglikelihood, word_count)
        _, word_count = result["word_perplexity"]
        # "short " = 1 word (scored text is "short ")
        assert word_count == 1

        # byte_perplexity tuple: (loglikelihood, byte_count)
        _, byte_count = result["byte_perplexity"]
        assert byte_count == 6  # len("short ".encode("utf-8"))

    def test_key_span_coverage_full(self):
        """All key spans within scored range => coverage = 1.0."""
        ll = FakeRichLoglikelihood(
            -3.0,
            losses=[1.0, 1.5, 0.5],
            offset_mapping=[(0, 5), (5, 10), (10, 15)],
        )
        doc = {
            "text": "abcdefghijklmno",
            "key_char_spans": [[0, 5], [10, 15]],
        }
        result = process_results(doc, [ll])
        assert result["key_span_coverage"] == 1.0

    def test_key_span_coverage_partial(self):
        """Some key spans beyond scored range => coverage < 1.0."""
        ll = FakeRichLoglikelihood(
            -3.0,
            losses=[1.0, 1.5],
            offset_mapping=[(0, 5), (5, 10)],
        )
        doc = {
            "text": "a" * 100,
            "key_char_spans": [[0, 5], [50, 60], [80, 90]],
        }
        result = process_results(doc, [ll])
        # Only span [0,5] is within scored_end=10
        assert result["key_span_coverage"] == pytest.approx(1 / 3)

    def test_no_key_spans_coverage_default(self):
        """No key spans => coverage defaults to 1.0."""
        ll = FakeRichLoglikelihood(
            -3.0,
            losses=[1.0],
            offset_mapping=[(0, 5)],
        )
        doc = {"text": "hello", "key_char_spans": []}
        result = process_results(doc, [ll])
        assert result["key_span_coverage"] == 1.0

    def test_zero_scored_tokens(self):
        """Empty losses/offset_mapping => coverage 0, safe denominators.

        Regression: scored_text previously fell back to full text when
        scored_end==0, corrupting perplexity denominators.  Denominators
        are clamped to 1 to prevent ZeroDivisionError in lm-eval's
        weighted_perplexity aggregator.
        """
        ll = FakeRichLoglikelihood(
            0.0,
            losses=[],
            offset_mapping=[],
        )
        doc = {
            "text": "a" * 1000,
            "key_char_spans": [[0, 100], [500, 600]],
        }
        result = process_results(doc, [ll])
        # Coverage must be 0.0, not 1.0
        assert result["key_span_coverage"] == 0.0
        # Perplexity denominators clamped to 1 (not full text length)
        _, word_count = result["word_perplexity"]
        assert word_count == 1
        _, byte_count = result["byte_perplexity"]
        assert byte_count == 1
        # LongPPL should be NaN (no key tokens scored)
        assert math.isnan(result["mean_key_loss"])


class TestLongpplAgg:
    """Test token-count-weighted LongPPL aggregation."""

    def test_weighted_formula(self):
        """exp(sum(n_d * L_d) / sum(n_d)) with known values."""
        items = [
            (2.0, 10),  # doc1: mean_loss=2.0, 10 key tokens
            (3.0, 20),  # doc2: mean_loss=3.0, 20 key tokens
        ]
        # Expected: exp((2.0*10 + 3.0*20) / (10+20)) = exp(80/30)
        expected = math.exp(80 / 30)
        assert longppl_agg(items) == pytest.approx(expected)

    def test_single_document(self):
        """Single document: exp(mean_loss)."""
        items = [(1.5, 5)]
        assert longppl_agg(items) == pytest.approx(math.exp(1.5))

    def test_nan_items_skipped(self):
        """NaN documents are excluded from aggregation."""
        items = [
            (2.0, 10),
            (float("nan"), 0),
            (3.0, 5),
        ]
        expected = math.exp((2.0 * 10 + 3.0 * 5) / 15)
        assert longppl_agg(items) == pytest.approx(expected)

    def test_all_nan_returns_nan(self):
        items = [(float("nan"), 0), (float("nan"), 0)]
        assert math.isnan(longppl_agg(items))

    def test_empty_returns_nan(self):
        assert math.isnan(longppl_agg([]))

    def test_equal_weighting_when_same_count(self):
        """When all docs have same token count, equals exp(mean of means)."""
        items = [(1.0, 5), (3.0, 5)]
        expected = math.exp((1.0 + 3.0) / 2)
        assert longppl_agg(items) == pytest.approx(expected)


class TestNanmean:
    def test_normal_values(self):
        assert nanmean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_with_nans(self):
        assert nanmean([1.0, float("nan"), 3.0]) == pytest.approx(2.0)

    def test_all_nans(self):
        assert math.isnan(nanmean([float("nan"), float("nan")]))

    def test_empty(self):
        assert math.isnan(nanmean([]))
