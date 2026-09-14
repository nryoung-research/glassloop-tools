"""Finite greedy-trace checks through the model's own ``generate`` method.

This module does not load a model, download anything, change devices, or launch
training. Scores come from HF generate's processed ``scores`` output, not raw
teacher-forced logits. Identity strings are caller-supplied pins, not attestation.
Only single-sequence greedy, EOS-terminated causal-LM generation is supported.
The score budget covers retained scores only; KV caches/model memory need a
separate profile. A certificate applies to this observed finite generation only.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any

import torch


class DecoderGuardError(RuntimeError):
    """An unsupported decoder configuration or a failed finite trace check."""


@dataclass(frozen=True)
class GreedyTrace:
    prompt_ids: tuple[int, ...]
    attention_mask: tuple[int, ...] | None
    completion_ids: tuple[int, ...]
    margins: tuple[float, ...]
    floors: tuple[float, ...]
    eos_token_ids: tuple[int, ...]
    termination_reason: str
    inherited_config_json: str
    max_new_tokens: int
    retained_score_bytes: int
    external_model_identity: str


@dataclass(frozen=True)
class TraceVerification:
    trace: GreedyTrace
    root_external_model_identity: str
    minimum_margin_surplus: float


def _token_tuple(value: torch.Tensor, name: str) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 1:
        raise DecoderGuardError(f"{name} must be a [1, length] tensor")
    if value.shape[1] == 0 or value.dtype not in (torch.int32, torch.int64):
        raise DecoderGuardError(f"{name} must contain nonempty integer tokens")
    return tuple(int(x) for x in value.detach().cpu().tolist()[0])


def _configuration(model: Any, generation_config: Any, cap: int) -> tuple[Any, str, tuple[int, ...], int]:
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise DecoderGuardError("max_new_tokens must be a positive integer")
    if getattr(model, "training", True):
        raise DecoderGuardError("model must already be in evaluation mode")
    source = generation_config if generation_config is not None else getattr(model, "generation_config", None)
    if source is None or not callable(getattr(source, "to_dict", None)):
        raise DecoderGuardError("a bound GenerationConfig with to_dict() is required")
    config = copy.deepcopy(source)
    # Ask HF itself to resolve model/default inheritance. Recent Transformers
    # releases leave unspecified fields as None until this preparation stage.
    # This is config preparation only, not an alternate decoding implementation.
    prepare = getattr(model, "_prepare_generation_config", None)
    controls = {"max_new_tokens": cap, "output_scores": True, "return_dict_in_generate": True}
    if callable(prepare):
        try:
            config, remaining = prepare(config, **controls)
        except Exception as exc:
            raise DecoderGuardError("HF GenerationConfig preparation failed") from exc
        if remaining:
            raise DecoderGuardError("unbound model kwargs remain after GenerationConfig preparation")
    else:
        # Used by protocol test doubles. Real HF models provide the helper.
        for key, value in controls.items():
            setattr(config, key, value)
    if getattr(config, "do_sample", False) is not False:
        raise DecoderGuardError("sampling is unsupported")
    if getattr(config, "num_beams", 1) != 1 or getattr(config, "num_beam_groups", 1) not in (None, 1):
        raise DecoderGuardError("beam decoding is unsupported")
    if getattr(config, "num_return_sequences", 1) != 1:
        raise DecoderGuardError("multiple return sequences are unsupported")
    mode_getter = getattr(config, "get_generation_mode", None)
    if callable(mode_getter) and mode_getter() != "greedy_search":
        raise DecoderGuardError("GenerationConfig does not select ordinary greedy search")
    if getattr(config, "penalty_alpha", None) not in (None, 0, 0.0):
        raise DecoderGuardError("contrastive decoding is unsupported")
    if getattr(config, "stop_strings", None):
        raise DecoderGuardError("stop-string termination is unsupported; this bridge certifies EOS only")
    if getattr(config, "max_time", None) is not None:
        raise DecoderGuardError("wall-clock stopping is unsupported")
    if any(bool(getattr(config, field, False)) for field in ("output_logits", "output_attentions", "output_hidden_states")):
        raise DecoderGuardError("extra retained model outputs are outside the score-storage budget")
    if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
        raise DecoderGuardError("only causal language models are supported")
    vocab = getattr(getattr(model, "config", None), "vocab_size", None)
    if not isinstance(vocab, int) or vocab < 2:
        raise DecoderGuardError("model.config.vocab_size must bind the complete score vocabulary")
    eos = getattr(config, "eos_token_id", None)
    eos = [eos] if isinstance(eos, int) and not isinstance(eos, bool) else eos
    if not isinstance(eos, (list, tuple)) or not eos:
        raise DecoderGuardError("at least one explicit EOS token ID is required")
    if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < vocab for x in eos):
        raise DecoderGuardError("invalid EOS token IDs")
    try:
        config_json = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DecoderGuardError("GenerationConfig is not serializable for exact comparison") from exc
    return config, config_json, tuple(dict.fromkeys(eos)), vocab


def _observe(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    external_model_identity: str,
    max_new_tokens: int,
    max_score_bytes: int,
    generation_config: Any,
) -> GreedyTrace:
    if not isinstance(external_model_identity, str) or not external_model_identity.strip():
        raise DecoderGuardError("an explicit external model identity is required (not verified here)")
    prompt = _token_tuple(input_ids, "input_ids")
    mask = None if attention_mask is None else _token_tuple(attention_mask, "attention_mask")
    if mask is not None and (len(mask) != len(prompt) or any(x not in (0, 1) for x in mask)):
        raise DecoderGuardError("attention_mask must match the prompt and contain only zero/one")
    config, config_json, eos, vocab = _configuration(model, generation_config, max_new_tokens)
    if any(not 0 <= token < vocab for token in prompt):
        raise DecoderGuardError("prompt token lies outside the model vocabulary")
    if isinstance(max_score_bytes, bool) or not isinstance(max_score_bytes, int) or max_score_bytes <= 0:
        raise DecoderGuardError("max_score_bytes must be a positive integer")
    # HF commonly returns float32 processed scores even for BF16 models. Unknown
    # dtype is budgeted as float64. This excludes caches and temporary activations.
    dtype = getattr(model, "dtype", None)
    item_bytes = 8 if dtype == torch.float64 or not isinstance(dtype, torch.dtype) else 4
    if max_new_tokens * vocab * item_bytes > max_score_bytes:
        raise DecoderGuardError("retained full-vocabulary scores exceed the declared preflight byte budget")
    kwargs = {"input_ids": input_ids.detach().clone(), "generation_config": config}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.detach().clone()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    sequences = getattr(output, "sequences", None)
    if not isinstance(sequences, torch.Tensor):
        raise DecoderGuardError("generate returned no bound sequences tensor")
    sequence = _token_tuple(sequences, "generated sequences")
    if sequence[: len(prompt)] != prompt:
        raise DecoderGuardError("generated sequence does not retain the exact raw prompt IDs")
    completion = sequence[len(prompt) :]
    scores = getattr(output, "scores", None)
    if not isinstance(scores, (tuple, list)) or not scores:
        raise DecoderGuardError("generate returned no bound processed score steps")
    if not completion or len(scores) != len(completion) or len(completion) > max_new_tokens:
        raise DecoderGuardError("missing or excess score steps for generated completion")
    if completion[-1] not in eos:
        raise DecoderGuardError("incomplete trace: no terminating EOS within the declared cap")
    if any(token in eos for token in completion[:-1]):
        raise DecoderGuardError("generated tokens continue after an EOS")
    margins: list[float] = []
    retained_bytes = 0
    for index, (token, score) in enumerate(zip(completion, scores)):
        if not isinstance(score, torch.Tensor) or score.shape != (1, vocab) or not score.is_floating_point():
            raise DecoderGuardError(f"step {index}: scores must bind the complete [1, vocab_size] vocabulary")
        if not 0 <= token < vocab:
            raise DecoderGuardError(f"step {index}: generated token is outside the vocabulary")
        retained_bytes += score.numel() * score.element_size()
        if retained_bytes > max_score_bytes:
            raise DecoderGuardError("actual retained score bytes exceed the declared budget")
        row = score.detach()[0]
        # -inf is a legitimate token mask. NaN/+inf cannot certify selection.
        if bool(torch.isnan(row).any()) or bool(torch.isposinf(row).any()):
            raise DecoderGuardError(f"step {index}: NaN or positive-infinite selection scores")
        winner = row[token]
        if not bool(torch.isfinite(winner)) or int(torch.argmax(row).item()) != token:
            raise DecoderGuardError(f"step {index}: returned token is not the processed-score winner")
        # Use every competitor; do not use a frozen top-k competitor list.
        rivals = torch.cat((row[:token], row[token + 1 :]))
        margin = float((winner - rivals.max()).item())
        if not margin > 0:
            raise DecoderGuardError(f"step {index}: winner has a tie or nonpositive margin")
        margins.append(margin)
    return GreedyTrace(
        prompt, mask, completion, tuple(margins), (), eos,
        f"eos:{completion[-1]}", config_json, max_new_tokens,
        retained_bytes, external_model_identity,
    )


def freeze_root_trace(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    external_model_identity: str,
    max_new_tokens: int = 256,
    floor_fraction: float = 0.1,
    floor_cap: float = 0.01,
    max_score_bytes: int = 256 * 1024 * 1024,
    generation_config: Any = None,
) -> GreedyTrace:
    """Observe complete processed scores and freeze capped root-relative floors.

    Inherited repetition penalties, suppression and both/list EOS IDs are passed
    intact to generate. Correctness of the returned code is the caller's job.
    A supplied config must represent the declared deployment config; the bridge
    records it but cannot verify external runtime/artifact identity claims.
    """
    if not math.isfinite(floor_fraction) or not 0 < floor_fraction <= 1:
        raise DecoderGuardError("floor_fraction must lie in (0, 1]")
    if not math.isfinite(floor_cap) or floor_cap <= 0:
        raise DecoderGuardError("floor_cap must be positive and finite")
    trace = _observe(
        model, input_ids, attention_mask, external_model_identity=external_model_identity,
        max_new_tokens=max_new_tokens, max_score_bytes=max_score_bytes,
        generation_config=generation_config,
    )
    floors = tuple(min(floor_fraction * margin, floor_cap) for margin in trace.margins)
    return GreedyTrace(**{**trace.__dict__, "floors": floors})


def verify_root_trace(
    model: Any,
    root: GreedyTrace,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    external_model_identity: str,
    max_score_bytes: int = 256 * 1024 * 1024,
    generation_config: Any = None,
) -> TraceVerification:
    """Run the same greedy decoder and require exact IDs and every root floor.

    No teacher forcing, external KV cache, adapter, or alternate decoder is supplied
    by this bridge. Cold-load identity verification remains external work.
    """
    if _token_tuple(input_ids, "input_ids") != root.prompt_ids:
        raise DecoderGuardError("candidate raw prompt differs from frozen root")
    mask = None if attention_mask is None else _token_tuple(attention_mask, "attention_mask")
    if mask != root.attention_mask:
        raise DecoderGuardError("candidate attention mask differs from frozen root")
    _, config_json, eos, _ = _configuration(model, generation_config, root.max_new_tokens)
    if config_json != root.inherited_config_json or eos != root.eos_token_ids:
        raise DecoderGuardError("candidate GenerationConfig differs from frozen root")
    if len(root.floors) != len(root.completion_ids) or not root.floors:
        raise DecoderGuardError("root has no complete frozen floor bank")
    if any(not math.isfinite(floor) or floor <= 0 for floor in root.floors):
        raise DecoderGuardError("invalid frozen root margin floors")
    observed = _observe(
        model, input_ids, attention_mask, external_model_identity=external_model_identity,
        max_new_tokens=root.max_new_tokens, max_score_bytes=max_score_bytes,
        generation_config=generation_config,
    )
    if observed.completion_ids != root.completion_ids:
        raise DecoderGuardError("candidate completion IDs differ from frozen root")
    surplus = min(margin - floor for margin, floor in zip(observed.margins, root.floors))
    if surplus < 0:
        raise DecoderGuardError("candidate violates a frozen root margin floor")
    return TraceVerification(observed, root.external_model_identity, surplus)
