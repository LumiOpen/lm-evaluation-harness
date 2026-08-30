"""
Custom lm-eval model for key-token LongPPL evaluation.

Extends HFLM to compute per-token losses during loglikelihood_rolling.
The standard HFLM only returns aggregate loglikelihood — this subclass
returns a RichLoglikelihood (a float subclass) that also carries
per-token losses and offset_mapping as attributes.

process_results accesses these via getattr() and raises if they
are missing (e.g. wrong --model or cached plain float).

Scoring alignment: The LongPPL reference implementation uses ``i-1``
index shifting to align losses with token positions.  This model
instead prepends ``prefix_token_id`` before document tokens, which
naturally aligns ``losses[i]`` with ``offset_mapping[i]`` — no index
shifting needed.  Both approaches score every document token including
the first, but they are not byte-for-byte identical.

Cache contract: Do not share ``--use_cache`` databases between
longppl_hf and standard hf model runs.  If a cached result is a plain
float (missing .losses/.offset_mapping), process_results will raise.
Clear the cache DB or omit ``--use_cache`` to resolve.

Usage:
    python -m lm_eval --model longppl_hf \
        --model_args pretrained=meta-llama/Llama-3.1-8B,max_length=32768,allow_truncation=true \
        --tasks crosslingual_longppl_fi

    # For models with short native context (e.g. 8K) evaluated at longer
    # lengths, enable auto RoPE scaling:
    python -m lm_eval --model longppl_hf \
        --model_args pretrained=LumiOpen/Poro-8B,max_length=32768,auto_rope_scaling=true,allow_truncation=true \
        --tasks crosslingual_longppl_fi
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

eval_logger = logging.getLogger(__name__)


class RichLoglikelihood(float):
    """A float that also carries per-token loss data.

    Behaves exactly like a float for all arithmetic, comparisons, and
    serialization — lm-eval's internals (caching, aggregation, JSON
    export) see an ordinary number.  But process_results can retrieve
    the attached per-token data via getattr().

    If the value comes from lm-eval's result cache (a plain float),
    getattr(..., None) returns None and process_results raises
    RuntimeError.  Omit ``--use_cache`` or clear the cache DB.
    """

    def __new__(
        cls,
        value: float,
        *,
        losses: list[float] | None = None,
        offset_mapping: list[tuple[int, int]] | None = None,
    ) -> RichLoglikelihood:
        obj = super().__new__(cls, value)
        obj.losses = losses
        obj.offset_mapping = offset_mapping
        return obj


@register_model("longppl_hf")
class LongPPLHFModel(HFLM):
    """HFLM subclass that returns per-token losses for key-token LongPPL.

    Differences from base HFLM:

    * ``_create_model`` optionally applies dynamic NTK RoPE scaling when
      ``auto_rope_scaling=true`` is passed and ``max_length`` exceeds
      the model's native context window.
    * ``loglikelihood_rolling`` does a single forward pass per document
      and returns :class:`RichLoglikelihood` values that carry per-token
      losses and offset_mapping alongside the aggregate loglikelihood.

    .. note::

       Multi-process data parallelism (``accelerate launch``) is **not**
       supported.  Use ``parallelize=True`` in ``--model_args`` for
       model-parallel inference on multiple GPUs instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Pop custom args before super().__init__() passes them through
        # to from_pretrained() where they'd be unrecognized.
        raw = kwargs.pop("auto_rope_scaling", False)
        self._auto_rope_scaling = (
            str(raw).lower() in ("true", "1", "yes")
            if isinstance(raw, str)
            else bool(raw)
        )

        raw_trunc = kwargs.pop("allow_truncation", False)
        self._allow_truncation = (
            str(raw_trunc).lower() in ("true", "1", "yes")
            if isinstance(raw_trunc, str)
            else bool(raw_trunc)
        )

        # Stash max_length before super().__init__() consumes it —
        # needed in _create_model to decide on RoPE scaling.
        # Depends on HFLM calling _get_config() before _create_model().
        self._requested_max_length = kwargs.get("max_length")

        super().__init__(*args, **kwargs)

        if self.AUTO_MODEL_CLASS is not None:
            import transformers

            if transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS:
                raise NotImplementedError(
                    "LongPPLHFModel only supports causal (decoder-only) "
                    "models.  Seq2seq models are not supported."
                )

        # Fast (Rust) tokenizers support offset_mapping; slow ones don't.
        # Without offset_mapping, key-token mapping is impossible.
        if not self.tokenizer.is_fast:
            raise NotImplementedError(
                f"LongPPLHFModel requires a fast (Rust) tokenizer for "
                f"offset_mapping support, but got {type(self.tokenizer).__name__}. "
                f"Pass use_fast_tokenizer=True in --model_args or use a "
                f"model with a fast tokenizer."
            )

    def _create_model(self, pretrained: str, **kwargs) -> None:
        """Optionally inject dynamic RoPE scaling before loading the model.

        Only active when ``auto_rope_scaling=true`` is passed in
        ``--model_args``.  At this point ``self._config`` is already set
        by ``_get_config()``, so we can check
        ``max_position_embeddings`` without an extra
        ``AutoConfig.from_pretrained`` call.

        Also validates that ``max_length`` does not exceed the model's
        native context window unless a valid scaling path is active.
        """
        native_ctx = getattr(self._config, "max_position_embeddings", None)
        max_length = self._requested_max_length
        exceeds_native = native_ctx and max_length and int(max_length) > native_ctx

        if self._auto_rope_scaling and exceeds_native:
            # Verify the model actually supports RoPE scaling
            has_rope = hasattr(self._config, "rope_scaling") or hasattr(
                self._config, "rope_theta"
            )
            if not has_rope:
                raise NotImplementedError(
                    f"auto_rope_scaling=true but {pretrained} does not "
                    f"appear to use RoPE (no rope_scaling or rope_theta "
                    f"in config).  RoPE scaling is not supported for "
                    f"this architecture.  Remove auto_rope_scaling or "
                    f"reduce max_length to {native_ctx}."
                )
            factor = int(max_length) / native_ctx
            kwargs["rope_scaling"] = {"type": "dynamic", "factor": factor}
            eval_logger.info(
                "auto_rope_scaling: applying dynamic NTK RoPE "
                "(factor=%.1f) to extend %d -> %s",
                factor,
                native_ctx,
                max_length,
            )
        elif exceeds_native and not self._auto_rope_scaling:
            raise ValueError(
                f"max_length={max_length} exceeds model's native "
                f"context window ({native_ctx}).  Either reduce "
                f"max_length to {native_ctx}, or pass "
                f"auto_rope_scaling=true if the model supports RoPE."
            )

        super()._create_model(pretrained, **kwargs)

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        """Single forward pass per document, returning per-token losses.

        Prepends ``prefix_token_id`` (matching HFLM rolling semantics) so
        that the first document token is scored.  For documents shorter
        than ``max_length`` the aggregate loglikelihood matches HFLM.

        .. warning::

           Unlike HFLM's rolling implementation, this does NOT use
           windowed passes for documents exceeding ``max_length``.
           By default, documents that exceed ``max_length - 1`` tokens
           raise ``ValueError``.  Pass ``allow_truncation=true`` in
           ``--model_args`` to truncate instead.
        """
        if self.world_size > 1:
            raise NotImplementedError(
                "LongPPLHFModel does not support multi-process data "
                "parallelism (accelerate launch).  Use parallelize=True "
                "in --model_args for model-parallel inference instead."
            )

        loglikelihoods: list[float] = []
        # Reserve one position for the prefix token
        max_doc_tokens = self.max_length - 1

        for req in tqdm(
            requests,
            disable=(disable_tqdm or self.rank != 0),
            desc="longppl_rolling",
        ):
            (text,) = req.args

            # Single tokenizer call for both token ids and offset_mapping,
            # using the same add_special_tokens logic as tok_encode.
            encoding = self.tokenizer(
                text,
                add_special_tokens=False or self.add_bos_token,
                truncation=self._allow_truncation,
                max_length=max_doc_tokens if self._allow_truncation else None,
                return_offsets_mapping=True,
            )
            token_ids: list[int] = encoding["input_ids"]
            offset_mapping: list[tuple[int, int]] = encoding["offset_mapping"]

            if len(token_ids) > max_doc_tokens:
                raise ValueError(
                    f"Document ({len(token_ids)} tokens) exceeds "
                    f"max_length ({self.max_length}).  This model does "
                    f"not use windowed rolling passes.  Pass "
                    f"allow_truncation=true in --model_args to truncate, "
                    f"or increase max_length."
                )

            # Prepend prefix token so the first document token is scored,
            # matching HFLM's rolling window semantics.
            input_ids = torch.tensor(
                [[self.prefix_token_id] + token_ids],
                dtype=torch.long,
                device=self.device,
            )

            # Forward pass via _model_call (handles autocast,
            # mixed_precision_dtype, torch.no_grad).
            logits = self._model_call(input_ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # per_token_nll[i] = NLL of predicting document token i
            per_token_nll = (
                F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                )
                .cpu()
                .tolist()
            )

            total_ll = RichLoglikelihood(
                -sum(per_token_nll),
                losses=per_token_nll,
                offset_mapping=offset_mapping,
            )
            loglikelihoods.append(total_ll)

            self.cache_hook.add_partial("loglikelihood_rolling", (text,), total_ll)

        return loglikelihoods
