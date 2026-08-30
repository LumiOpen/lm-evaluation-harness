"""Tests for longppl_hf model (LongPPLHFModel)."""

from __future__ import annotations

import os

import numpy as np
import pytest


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TestLongPPLHFModel:
    """Test LongPPLHFModel initialization and loglikelihood_rolling."""

    @pytest.fixture(scope="class")
    def model(self):
        from lm_eval.models.longppl_hf import LongPPLHFModel

        return LongPPLHFModel(
            pretrained="sshleifer/tiny-gpt2",
            device="cpu",
            dtype="float32",
        )

    @pytest.fixture(scope="class")
    def hflm(self):
        from lm_eval.models.huggingface import HFLM

        return HFLM(
            pretrained="sshleifer/tiny-gpt2",
            device="cpu",
            dtype="float32",
        )

    def test_loglikelihood_rolling_returns_rich_objects(self, model):
        """Results should carry .losses and .offset_mapping attributes."""
        from lm_eval.api.instance import Instance

        req = Instance(
            request_type="loglikelihood_rolling",
            doc={},
            arguments=("The cat sat on the mat.",),
            idx=0,
        )
        results = model.loglikelihood_rolling([req])
        assert len(results) == 1

        result = results[0]
        assert hasattr(result, "losses")
        assert hasattr(result, "offset_mapping")
        assert result.losses is not None
        assert result.offset_mapping is not None
        assert len(result.losses) > 0
        assert len(result.offset_mapping) > 0

    def test_loglikelihood_parity_with_hflm(self, model, hflm):
        """Aggregate loglikelihood should match HFLM within tolerance."""
        from lm_eval.api.instance import Instance

        text = "The quick brown fox jumps over the lazy dog."
        req = Instance(
            request_type="loglikelihood_rolling",
            doc={},
            arguments=(text,),
            idx=0,
        )
        longppl_result = model.loglikelihood_rolling([req])
        hflm_result = hflm.loglikelihood_rolling([req])

        assert np.allclose(
            float(longppl_result[0]),
            float(hflm_result[0]),
            atol=1e-3,
        ), (
            f"LongPPL ll={float(longppl_result[0]):.4f} vs "
            f"HFLM ll={float(hflm_result[0]):.4f}"
        )

    def test_losses_align_with_offset_mapping(self, model):
        """Number of losses should equal number of tokens (offset entries)."""
        from lm_eval.api.instance import Instance

        req = Instance(
            request_type="loglikelihood_rolling",
            doc={},
            arguments=("Hello world, this is a test.",),
            idx=0,
        )
        results = model.loglikelihood_rolling([req])
        result = results[0]

        # losses[i] corresponds to predicting token i
        # offset_mapping[i] gives char range for token i
        assert len(result.losses) == len(result.offset_mapping)

    def test_losses_are_positive(self, model):
        """NLL losses should be non-negative."""
        from lm_eval.api.instance import Instance

        req = Instance(
            request_type="loglikelihood_rolling",
            doc={},
            arguments=("Testing positive losses.",),
            idx=0,
        )
        results = model.loglikelihood_rolling([req])
        assert all(loss >= 0 for loss in results[0].losses)

    def test_aggregate_is_negative_sum_of_losses(self, model):
        """Float value should equal -sum(losses)."""
        from lm_eval.api.instance import Instance

        req = Instance(
            request_type="loglikelihood_rolling",
            doc={},
            arguments=("Test aggregation.",),
            idx=0,
        )
        results = model.loglikelihood_rolling([req])
        result = results[0]
        assert float(result) == pytest.approx(-sum(result.losses), abs=1e-4)


class TestLongPPLHFModelTruncation:
    """Test truncation behavior respects allow_truncation flag."""

    @pytest.fixture(scope="class")
    def strict_model(self):
        from lm_eval.models.longppl_hf import LongPPLHFModel

        return LongPPLHFModel(
            pretrained="sshleifer/tiny-gpt2",
            device="cpu",
            dtype="float32",
            max_length=10,  # very short to trigger truncation
        )

    @pytest.fixture(scope="class")
    def lenient_model(self):
        from lm_eval.models.longppl_hf import LongPPLHFModel

        return LongPPLHFModel(
            pretrained="sshleifer/tiny-gpt2",
            device="cpu",
            dtype="float32",
            max_length=10,
            allow_truncation="true",
        )

    def test_default_raises_on_overflow(self, strict_model):
        """Without allow_truncation, long docs raise ValueError."""
        from lm_eval.api.instance import Instance

        long_text = "word " * 100  # way more than 10 tokens
        req = Instance(
            request_type="loglikelihood_rolling",
            doc={},
            arguments=(long_text,),
            idx=0,
        )
        with pytest.raises(ValueError, match="exceeds max_length"):
            strict_model.loglikelihood_rolling([req])

    def test_allow_truncation_succeeds(self, lenient_model):
        """With allow_truncation=true, long docs are truncated."""
        from lm_eval.api.instance import Instance

        long_text = "word " * 100
        req = Instance(
            request_type="loglikelihood_rolling",
            doc={},
            arguments=(long_text,),
            idx=0,
        )
        results = lenient_model.loglikelihood_rolling([req])
        assert len(results) == 1
        # Should have fewer tokens than the full text
        assert len(results[0].losses) <= 9  # max_length=10 minus 1 prefix

    def test_short_doc_ok_without_flag(self, strict_model):
        """Documents within max_length work without allow_truncation."""
        from lm_eval.api.instance import Instance

        req = Instance(
            request_type="loglikelihood_rolling",
            doc={},
            arguments=("Hi",),
            idx=0,
        )
        results = strict_model.loglikelihood_rolling([req])
        assert len(results) == 1


class TestLongPPLHFModelScaling:
    """Test max_length and RoPE scaling validation."""

    def test_max_length_exceeds_native_without_scaling_raises(self):
        """max_length > native context without auto_rope_scaling raises."""
        from lm_eval.models.longppl_hf import LongPPLHFModel

        with pytest.raises(ValueError, match="exceeds model's native"):
            LongPPLHFModel(
                pretrained="sshleifer/tiny-gpt2",
                device="cpu",
                dtype="float32",
                max_length=32768,  # tiny-gpt2 has 1024 native
            )

    def test_auto_rope_scaling_on_non_rope_model_raises(self):
        """auto_rope_scaling=true on a non-RoPE model raises cleanly."""
        from lm_eval.models.longppl_hf import LongPPLHFModel

        with pytest.raises(NotImplementedError, match="does not.*use RoPE"):
            LongPPLHFModel(
                pretrained="sshleifer/tiny-gpt2",
                device="cpu",
                dtype="float32",
                max_length=32768,
                auto_rope_scaling="true",
            )


class TestLongPPLHFModelRejection:
    """Test that invalid configurations are rejected early."""

    def test_seq2seq_model_rejected(self):
        """Seq2seq models should raise NotImplementedError."""
        from lm_eval.models.longppl_hf import LongPPLHFModel

        with pytest.raises(NotImplementedError, match="causal"):
            LongPPLHFModel(
                pretrained="sshleifer/tiny-marian-en-de",
                device="cpu",
                dtype="float32",
            )

    def test_slow_tokenizer_rejected(self):
        """Slow tokenizer should raise NotImplementedError, not just warn."""
        from lm_eval.models.longppl_hf import LongPPLHFModel

        with pytest.raises(NotImplementedError, match="fast.*tokenizer"):
            LongPPLHFModel(
                pretrained="sshleifer/tiny-gpt2",
                device="cpu",
                dtype="float32",
                use_fast_tokenizer=False,
            )
