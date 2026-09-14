"""Differentiable decision margins for SEARCH, never a finite certificate.

The caller owns a connected functional forward over native/master parameters.
HF supplies the actual configured deterministic logits-processor chain. A full
prefix forward can differ numerically from cached generate; decoder.py's actual
generate check remains required. No models, checkpoints or devices are created
by importing this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import torch

from decoder import DecoderGuardError, GreedyTrace, _configuration


class SearchMarginError(RuntimeError):
    """The requested surrogate is unsupported or cannot supply a finite row."""


@dataclass(frozen=True)
class DecisionPrefix:
    ids: tuple[int, ...]
    attention_mask: tuple[int, ...] | None
    expected_token_id: int
    generation_position: int
    original_prompt_length: int
    floor: float


def decision_prefix(root: GreedyTrace, position: int) -> DecisionPrefix:
    """Map a threatened root position to its exact raw prompt + prior tokens."""
    if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < len(root.completion_ids):
        raise SearchMarginError("position is outside the frozen completion")
    if len(root.floors) != len(root.completion_ids):
        raise SearchMarginError("root has no complete frozen floor bank")
    ids = root.prompt_ids + root.completion_ids[:position]
    mask = None if root.attention_mask is None else root.attention_mask + (1,) * position
    return DecisionPrefix(ids, mask, root.completion_ids[position], position, len(root.prompt_ids), root.floors[position])


# Stateful guidance, watermarking and arbitrary custom processors are deliberately
# outside this bounded search bridge. These standard deterministic HF processors
# operate on the supplied token IDs and a connected score tensor.
_SUPPORTED_PROCESSORS = frozenset({
    "RepetitionPenaltyLogitsProcessor", "EncoderRepetitionPenaltyLogitsProcessor",
    "SequenceBiasLogitsProcessor", "NoRepeatNGramLogitsProcessor",
    "EncoderNoRepeatNGramLogitsProcessor", "NoBadWordsLogitsProcessor",
    "MinLengthLogitsProcessor", "MinNewTokensLengthLogitsProcessor",
    "ForcedBOSTokenLogitsProcessor", "ForcedEOSTokenLogitsProcessor",
    "InfNanRemoveLogitsProcessor", "ExponentialDecayLengthPenalty",
    "SuppressTokensLogitsProcessor", "SuppressTokensAtBeginLogitsProcessor",
    "LogitNormalization",
})


def _processors(model: Any, config: Any, root: GreedyTrace, device: torch.device):
    if getattr(config, "guidance_scale", None) not in (None, 1, 1.0):
        raise SearchMarginError("classifier-free guidance is outside the search bridge")
    if getattr(config, "watermarking_config", None) is not None:
        raise SearchMarginError("watermark processors are outside the search bridge")
    special = getattr(model, "_prepare_special_tokens", None)
    lengths = getattr(model, "_prepare_generated_length", None)
    build = getattr(model, "_get_logits_processor", None)
    if not all(callable(item) for item in (special, lengths, build)):
        raise SearchMarginError("HF generation preparation APIs are required")
    prompt = torch.tensor([root.prompt_ids], dtype=torch.long, device=device)
    try:
        # These HF APIs mutate only the already copied configuration. Length
        # offsets must use the ORIGINAL prompt, not the growing decision prefix.
        special(config, kwargs_has_attention_mask=root.attention_mask is not None, device=device)
        config = lengths(
            generation_config=config, has_default_max_length=True,
            has_default_min_length=True, model_input_name="input_ids",
            input_ids_length=len(root.prompt_ids), inputs_tensor=prompt,
        )
        chain = build(
            generation_config=config, input_ids_seq_length=len(root.prompt_ids),
            encoder_input_ids=prompt, prefix_allowed_tokens_fn=None,
            logits_processor=None, device=device, model_kwargs={},
        )
    except Exception as exc:
        raise SearchMarginError("HF deterministic processor preparation failed") from exc
    for processor in chain:
        cls = type(processor)
        if cls.__module__ != "transformers.generation.logits_process" or cls.__name__ not in _SUPPORTED_PROCESSORS:
            raise SearchMarginError(f"unsupported search processor: {cls.__module__}.{cls.__name__}")
    return chain


def make_margin_callback(
    model: Any,
    root_trace: GreedyTrace,
    position: int,
    functional_forward: Callable[[torch.Tensor, torch.Tensor | None], Any],
    *,
    generation_config: Any = None,
    device: torch.device | str = "cpu",
    observed_margin: float | None = None,
) -> Callable[[], torch.Tensor]:
    """Return a zero-argument scalar callback for controller.MarginConstraint.

    ``functional_forward(prefix_ids, attention_mask)`` must return connected
    [1, prefix_length, vocab] logits, [1, vocab] next logits, or an HF output with
    such ``logits``. It must not detach/copy the active master path. This bridge
    verifies requires_grad, but the caller must verify which parameters it owns.

    If supplied, ``observed_margin`` must be a FRESH actual-generate margin at
    the current origin. The callback uses that scalar value and the surrogate's
    derivative. Rebuild it after an accepted update or constraint refresh.
    Otherwise the scalar is the teacher-forced/prefill surrogate margin itself.
    """
    if not callable(functional_forward):
        raise SearchMarginError("functional_forward must be callable")
    if observed_margin is not None and not math.isfinite(observed_margin):
        raise SearchMarginError("observed_margin must be finite when anchoring a search row")
    decision = decision_prefix(root_trace, position)
    device = torch.device(device)

    def margin() -> torch.Tensor:
        try:
            config, config_json, eos, vocab = _configuration(model, generation_config, root_trace.max_new_tokens)
        except DecoderGuardError as exc:
            raise SearchMarginError(str(exc)) from exc
        if config_json != root_trace.inherited_config_json or eos != root_trace.eos_token_ids:
            raise SearchMarginError("current GenerationConfig differs from the frozen root")
        prefix = torch.tensor([decision.ids], dtype=torch.long, device=device)
        mask = None if decision.attention_mask is None else torch.tensor([decision.attention_mask], dtype=torch.long, device=device)
        chain = _processors(model, config, root_trace, device)
        output = functional_forward(prefix, mask)
        logits = output if isinstance(output, torch.Tensor) else getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
            raise SearchMarginError("functional_forward must return floating connected logits")
        if logits.shape == (1, len(decision.ids), vocab):
            scores = logits[:, -1, :]
        elif logits.shape == (1, vocab):
            scores = logits
        else:
            raise SearchMarginError("functional logits do not bind the complete expected vocabulary")
        if scores.device != device:
            raise SearchMarginError("functional logits and frozen prefix are on different devices")
        if not scores.requires_grad:
            raise SearchMarginError("functional logits are detached; no master-connected search gradient")
        if not bool(torch.isfinite(scores).all()):
            raise SearchMarginError("nonfinite raw functional logits cannot define a trustworthy search row")
        # Standard HF greedy generation processes float32 next-token logits.
        # Clone because some processor implementations modify scores in place.
        processed = chain(prefix, scores.float().clone())
        token = decision.expected_token_id
        if processed.shape != (1, vocab) or bool(torch.isnan(processed).any()) or bool(torch.isposinf(processed).any()):
            raise SearchMarginError("invalid complete processed score row")
        winner = processed[0, token]
        rivals = torch.cat((processed[0, :token], processed[0, token + 1 :]))
        rival = rivals.max()
        if not bool(torch.isfinite(winner)):
            raise SearchMarginError("expected root token is masked; no finite gradient margin")
        if not bool(torch.isfinite(rival)):
            raise SearchMarginError("all rivals are masked; the forced decision needs no finite search row")
        result = winner - rival
        if not result.requires_grad or not bool(torch.isfinite(result)):
            raise SearchMarginError("processed margin is not a finite connected scalar")
        if observed_margin is not None:
            # Subtract identical values first so a small observed margin is not
            # rounded away when the surrogate's absolute value is very large.
            result = (result - result.detach()) + result.new_tensor(observed_margin)
        return result

    return margin
