"""One constrained native-parameter step with explicit finite-check callbacks.

This integrates optimizer displacement, the Gram projector, and transactions.
It does NOT load/export a checkpoint, construct datasets, or certify a decoder.
The finite checker must materialize and inspect the intended deployed function.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from gram_projector import project_gram
from transaction import TensorTransaction


@dataclass(frozen=True)
class MarginConstraint:
    name: str
    value: object  # zero-argument callable returning a scalar tensor
    floor: float


@dataclass(frozen=True)
class FiniteCheck:
    preserved: bool
    materialized_changed: bool
    detail: str


@dataclass(frozen=True)
class StepResult:
    accepted: bool
    reason: str
    attempts: int
    step_scale: float
    teaching_before: float
    teaching_after: float | None
    proposal_norm: float
    corrected_norm: float
    predicted_teaching_decrease: float


def _dot(left, right):
    # Chunked float64 products bound temporary memory without concatenating an
    # entire active window. These are CPU statistics, not retained GPU graphs.
    values = []
    for a, b in zip(left, right):
        x, y = a.reshape(-1), b.reshape(-1)
        if x.numel() != y.numel():
            raise ValueError('Direction dimensions disagree')
        for start in range(0, x.numel(), 262144):
            values.append(float((x[start:start + 262144].double()
                                 * y[start:start + 262144].double()).sum()))
    return math.fsum(values)


def _scalar(value, name):
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(name + ' must return one scalar tensor')
    number = float(value.detach())
    if not math.isfinite(number):
        raise ValueError(name + ' is nonfinite')
    return number


def _gradient(value, parameters):
    values = torch.autograd.grad(value, parameters, allow_unused=True)
    if all(g is None for g in values):
        raise ValueError('Search loss/margin is disconnected from every editable parameter')
    return tuple(torch.zeros_like(p, device='cpu') if g is None
                 else g.detach().cpu().clone() for p, g in zip(parameters, values))


def constrained_step(named_parameters, optimizer, teaching_loss, *,
                     constraints, finite_check, finite_teaching_loss, optimization_loss=None,
                     optimization_backward=None,
                     radius_multiplier=1., descent_fraction=.05,
                     minimum_loss_decrease=1e-6, step_scales=(1., .5, .25, .125),
                     capture_cuda_devices=()):
    """Propose Adam/SGD displacement; correct jointly, finite-check, then commit.

    Loss callbacks must use identical fixed token examples/masks in deterministic
    evaluation mode. optimization_loss may add declared replay to teaching.
    optimization_backward is an alternative memory-bounded callback that calls
    backward one sequence at a time and returns a finite detached loss total;
    it is mutually exclusive with optimization_loss. The controller zeroes grads.
    teaching_loss supplies a differentiable acquisition proxy; the REQUIRED
    finite_teaching_loss callback measures that token set on the actual native
    materialization, returning a finite float. It must not silently return the
    FP32-master proxy loss when BF16 deployment is the claimed target.
    At most four current margin callbacks plus one acquisition row are used.
    The finite callback must check the ENTIRE protected bank, all rivals, native
    representation and runtime, not just these four local search constraints.
    All backtracked proposals use the same pre-step parameter origin. The caller
    must count attempted exposures/forwards; this routine does not own a dataset.
    """
    named = tuple(named_parameters.items()) if isinstance(named_parameters, dict) else tuple(named_parameters)
    parameters = tuple(p for _, p in named)
    constraints = tuple(constraints)
    if optimization_loss is not None and optimization_backward is not None:
        raise ValueError('Choose optimization_loss or sequence-wise optimization_backward')
    if len(constraints) > 4 or len({c.name for c in constraints}) != len(constraints):
        raise ValueError('At most four uniquely named preservation constraints')
    if any(not math.isfinite(c.floor) for c in constraints):
        raise ValueError('Nonfinite margin floor')
    if not 0 < descent_fraction <= 1 or not math.isfinite(radius_multiplier) or radius_multiplier <= 0:
        raise ValueError('Invalid descent fraction or radius multiplier')
    if not math.isfinite(minimum_loss_decrease) or minimum_loss_decrease <= 0:
        raise ValueError('A positive finite progress threshold is required')
    if (not step_scales or any(not math.isfinite(x) or not 0 < x <= 1 for x in step_scales)
            or any(a <= b for a, b in zip(step_scales, step_scales[1:]))):
        raise ValueError('Step scales must be finite, positive and strictly decreasing')

    before = math.nan
    with TensorTransaction(named, optimizer, torch_module=torch,
                           capture_cuda_devices=capture_cuda_devices) as tx:
        before = float(finite_teaching_loss())
        if not math.isfinite(before):
            raise ValueError('nonfinite original materialized teaching loss')
        value = teaching_loss()
        _scalar(value, 'teaching_loss')
        teaching_gradient = _gradient(value, parameters)
        rows, bounds = [], []
        for constraint in constraints:
            margin = constraint.value()
            initial = _scalar(margin, constraint.name)
            rows.append(_gradient(margin, parameters))
            bounds.append(constraint.floor - initial)

        optimizer.zero_grad(set_to_none=True)
        if optimization_backward is None:
            loss = (optimization_loss or teaching_loss)()
            _scalar(loss, 'optimization_loss')
            loss.backward()
        else:
            loss_total = float(optimization_backward())
            if not math.isfinite(loss_total):
                raise ValueError('optimization_backward returned a nonfinite total')
        optimizer.step()
        proposal = tuple((p.detach().cpu() - tx.original_weights[name].detach().cpu()).clone()
                         for name, p in named)
        tx.restore_weights_only()
        norm_squared = _dot(proposal, proposal)
        predicted_decrease = -_dot(teaching_gradient, proposal)
        norm = math.sqrt(max(0., norm_squared))
        if not norm or not math.isfinite(predicted_decrease) or predicted_decrease <= 0:
            return StepResult(False, 'proposal has no finite acquisition descent', 0, 0., before, None,
                              norm, 0., predicted_decrease)
        rows.append(tuple(-g for g in teaching_gradient))
        bounds.append(descent_fraction * predicted_decrease)
        gram = np.asarray([[_dot(a, b) for b in rows] for a in rows])
        a = np.asarray([_dot(row, proposal) for row in rows])
        projection = project_gram(gram, a, bounds, norm_squared, norm * radius_multiplier)
        if projection is None:
            return StepResult(False, 'no feasible local proposal found within the declared search', 0,
                              0., before, None, norm, 0., predicted_decrease)
        direction = []
        for index, u in enumerate(proposal):
            d = u * projection.proposal_scale
            for weight, row in zip(projection.row_weights, rows):
                d = d + weight * row[index]
            direction.append(d)
        # Recheck assembled finite-precision direction before applying it.
        values = np.asarray([_dot(row, direction) for row in rows])
        assembled_norm = math.sqrt(max(0., _dot(direction, direction)))
        if (np.any(values < np.asarray(bounds) - 1e-6 * np.maximum(1., np.abs(bounds)))
                or assembled_norm > norm * radius_multiplier * (1 + 1e-6)):
            return StepResult(False, 'assembled tensor direction fails numerical proposal checks', 0,
                              0., before, None, norm, assembled_norm, predicted_decrease)
        after, reason = None, 'no finite candidate accepted'
        for attempt, scale in enumerate(step_scales, 1):
            if scale * values[-1] < bounds[-1] * (1 - 1e-10):
                reason = 'backtracked step misses the fixed acquisition-descent floor'
                continue
            with torch.no_grad():
                for (name, p), delta in zip(named, direction):
                    p.copy_(tx.original_weights[name].to(device=p.device, dtype=p.dtype)
                            + scale * delta.to(device=p.device, dtype=p.dtype))
            after = float(finite_teaching_loss())
            if not math.isfinite(after):
                reason = 'nonfinite materialized teaching loss'
                continue
            if before - after <= minimum_loss_decrease:
                reason = 'finite acquisition progress is below the declared threshold'
                continue
            check = finite_check()
            if not isinstance(check, FiniteCheck):
                raise TypeError('finite_check must return FiniteCheck')
            if type(check.preserved) is not bool or type(check.materialized_changed) is not bool:
                raise TypeError('finite_check verdicts must be booleans')
            if not check.preserved or not check.materialized_changed:
                reason = check.detail or 'finite preservation or materialization check failed'
                continue
            tx.commit()
            return StepResult(True, check.detail, attempt, scale, before, after, norm,
                              assembled_norm * scale, values[-1] * scale)
        return StepResult(False, reason, len(step_scales), 0., before, after, norm,
                          assembled_norm, predicted_decrease)
