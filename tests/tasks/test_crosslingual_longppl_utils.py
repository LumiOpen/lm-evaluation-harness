"""Tests for crosslingual_longppl cal_overlap (containment semantics)."""

from lm_eval.tasks.crosslingual_longppl.utils import cal_overlap


class TestCalOverlap:
    """Verify cal_overlap uses containment: token must be fully inside a key span."""

    def test_token_fully_inside_span(self):
        """Token [5,10) inside span [0,20) => key."""
        offset_mapping = [(5, 10)]
        key_char_spans = [[0, 20]]
        assert cal_overlap(offset_mapping, key_char_spans) == [0]

    def test_token_exactly_matches_span(self):
        """Token [5,10) equals span [5,10) => key."""
        offset_mapping = [(5, 10)]
        key_char_spans = [[5, 10]]
        assert cal_overlap(offset_mapping, key_char_spans) == [0]

    def test_token_partially_overlaps_not_contained(self):
        """Token [5,15) overlaps span [0,10) but extends past it => NOT key."""
        offset_mapping = [(5, 15)]
        key_char_spans = [[0, 10]]
        assert cal_overlap(offset_mapping, key_char_spans) == []

    def test_token_starts_before_span(self):
        """Token [0,10) starts before span [5,20) => NOT key."""
        offset_mapping = [(0, 10)]
        key_char_spans = [[5, 20]]
        assert cal_overlap(offset_mapping, key_char_spans) == []

    def test_token_outside_span(self):
        """Token [20,25) after span [0,10) => NOT key."""
        offset_mapping = [(20, 25)]
        key_char_spans = [[0, 10]]
        assert cal_overlap(offset_mapping, key_char_spans) == []

    def test_empty_key_char_spans(self):
        offset_mapping = [(0, 5), (5, 10)]
        assert cal_overlap(offset_mapping, []) == []

    def test_empty_offset_mapping(self):
        key_char_spans = [[0, 10]]
        assert cal_overlap([], key_char_spans) == []

    def test_zero_width_tokens_skipped(self):
        """Zero-width tokens (start == end) are never key tokens."""
        offset_mapping = [(0, 0), (0, 5), (5, 5), (5, 10)]
        key_char_spans = [[0, 10]]
        assert cal_overlap(offset_mapping, key_char_spans) == [1, 3]

    def test_multiple_spans(self):
        """Tokens matched against multiple sorted, non-overlapping spans."""
        offset_mapping = [
            (0, 3),  # inside span [0,5)
            (3, 5),  # inside span [0,5)
            (5, 8),  # between spans => NOT key
            (10, 13),  # inside span [10,20)
            (15, 18),  # inside span [10,20)
            (20, 25),  # outside => NOT key
        ]
        key_char_spans = [[0, 5], [10, 20]]
        assert cal_overlap(offset_mapping, key_char_spans) == [0, 1, 3, 4]

    def test_token_straddles_span_boundary(self):
        """Token [4,8) straddles span [0,5) boundary => NOT key (not contained)."""
        offset_mapping = [(4, 8)]
        key_char_spans = [[0, 5]]
        assert cal_overlap(offset_mapping, key_char_spans) == []

    def test_overlapping_spans_token_in_second(self):
        """Token not contained in span A but contained in overlapping span B.

        Regression: two-pointer only checked span j, missing later
        overlapping spans.  Reproduced on dataset indices 306, 589, 637.
        """
        # Span A=[0,15], Span B=[5,20] — overlapping
        # Token [12,18] — not in A (extends past 15), but in B
        offset_mapping = [(12, 18)]
        key_char_spans = [[0, 15], [5, 20]]
        assert cal_overlap(offset_mapping, key_char_spans) == [0]

    def test_nested_spans(self):
        """Token contained in inner nested span."""
        # Outer=[0,30], Inner=[10,20]
        # Token [12,15] — contained in both
        offset_mapping = [(12, 15)]
        key_char_spans = [[0, 30], [10, 20]]
        assert cal_overlap(offset_mapping, key_char_spans) == [0]

    def test_overlapping_spans_token_only_in_first(self):
        """Token contained only in first of two overlapping spans."""
        # Span A=[0,15], Span B=[10,20]
        # Token [2,5] — in A, not in B
        offset_mapping = [(2, 5)]
        key_char_spans = [[0, 15], [10, 20]]
        assert cal_overlap(offset_mapping, key_char_spans) == [0]

    def test_overlapping_spans_multiple_tokens(self):
        """Multiple tokens across overlapping spans."""
        # Span A=[0,15], Span B=[10,25]
        offset_mapping = [
            (2, 5),  # in A only
            (12, 14),  # in A and B
            (16, 20),  # in B only
            (26, 30),  # in neither
        ]
        key_char_spans = [[0, 15], [10, 25]]
        assert cal_overlap(offset_mapping, key_char_spans) == [0, 1, 2]

    def test_many_tokens_few_spans(self):
        """Should handle many tokens efficiently."""
        offset_mapping = [(i, i + 1) for i in range(1000)]
        key_char_spans = [[100, 200], [500, 600]]
        result = cal_overlap(offset_mapping, key_char_spans)
        assert result == list(range(100, 200)) + list(range(500, 600))

    def test_reference_impl_alignment(self):
        """Verify our containment matches the LongPPL reference semantics.

        Reference uses: if a_start >= b_start and a_end <= b_end
        where a = token span, b = key span.
        """
        # Simulate a realistic tokenization scenario
        offset_mapping = [
            (0, 4),  # "The "
            (4, 8),  # "cat "
            (8, 11),  # "sat"
            (11, 12),  # " "
            (12, 14),  # "on"
            (14, 15),  # " "
            (15, 18),  # "the"
            (18, 19),  # " "
            (19, 22),  # "mat"
        ]
        # Key span covers "cat sat" = [4, 11]
        key_char_spans = [[4, 11]]
        result = cal_overlap(offset_mapping, key_char_spans)
        # Token "cat " [4,8) is inside [4,11) => key
        # Token "sat" [8,11) is inside [4,11) => key
        # Token " " [11,12) starts at span end => NOT inside
        assert result == [1, 2]
