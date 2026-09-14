"""Small constraint-space trust-region projection; no torch or model imports.

Solve min ||d-u||_2 subject to A d >= b and ||d||_2 <= radius, using only
G=A A^T, a=A u and ||u||^2. A caller can stream large parameter gradients
to form these small statistics and assemble d once from the returned weights.

The implementation enumerates at most 8 constraints. It is a numerical
proposal solver, not a rigorous infeasibility certificate or a model guard.
"""
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np


@dataclass(frozen=True)
class Projection:
    proposal_scale: float
    row_weights: tuple
    predicted_constraint_values: tuple
    delta_norm: float
    distance_to_proposal: float
    active_constraints: tuple
    ball_active: bool
    minimum_slack: float


def project_gram(gram, row_proposal_dot, bounds, proposal_norm_squared,
                 radius, *, tolerance=1e-9):
    """Return d=s*u+A.T*w coefficients, or None if no feasible candidate found.

    Constraints are normalized internally, so positive row rescaling preserves
    the problem. Near-rank-deficient systems use a pseudoinverse and residual
    checks. None is only failure of this bounded floating-point search; do not
    infer infeasibility of a full nonlinear writer from it.
    """
    g = np.asarray(gram, dtype=np.float64)
    a = np.asarray(row_proposal_dot, dtype=np.float64)
    b = np.asarray(bounds, dtype=np.float64)
    m = len(b)
    if m > 8 or g.shape != (m, m) or a.shape != (m,) or b.shape != (m,):
        raise ValueError('Require a square Gram matrix for at most eight constraints')
    if not all(np.all(np.isfinite(x)) for x in (g, a, b)):
        raise ValueError('All inputs must be finite')
    if not math.isfinite(radius) or radius < 0:
        raise ValueError('Radius must be finite and nonnegative')
    if not math.isfinite(proposal_norm_squared) or proposal_norm_squared < 0:
        raise ValueError('Proposal norm squared must be finite and nonnegative')
    if not math.isfinite(tolerance) or not 0 < tolerance < 1e-3:
        raise ValueError('Invalid numerical tolerance')
    if not np.allclose(g, g.T, rtol=tolerance, atol=tolerance):
        raise ValueError('Gram matrix must be symmetric')
    if m and np.any(np.diag(g) < 0):
        raise ValueError('Negative Gram diagonal')

    norms = np.sqrt(np.diag(g)) if m else np.empty(0)
    nonzero = norms > 0
    # A genuinely zero vector cannot have a nonzero inner product with u.
    if np.any(a[~nonzero] != 0) or (m and np.any(g[~nonzero] != 0)):
        raise ValueError('Inconsistent zero constraint vector')
    if np.any(b[~nonzero] > 0):
        return None
    if radius == 0:
        if np.any(b > 0):
            return None
        return Projection(0., tuple(0. for _ in range(m)), tuple(0. for _ in range(m)),
                          0., math.sqrt(proposal_norm_squared), (), True,
                          float((-b).min()) if m else math.inf)
    scales = np.where(nonzero, norms, 1.0)
    gn = g / scales[:, None] / scales[None, :]
    an = a / scales
    bn = b / scales
    u_norm = math.sqrt(proposal_norm_squared)
    u_scale = u_norm if u_norm else 1.0
    augmented = np.empty((m + 1, m + 1), dtype=np.float64)
    augmented[:m, :m] = gn
    augmented[:m, m] = an / u_scale
    augmented[m, :m] = an / u_scale
    augmented[m, m] = proposal_norm_squared / (u_scale * u_scale)
    if np.linalg.eigvalsh(augmented)[0] < -10 * tolerance:
        raise ValueError('Gram/proposal inner products are not positive semidefinite')

    best = None
    indices = tuple(int(i) for i in np.flatnonzero(nonzero))
    for count in range(len(indices) + 1):
        for active in combinations(indices, count):
            if active:
                ix = np.asarray(active)
                h = gn[np.ix_(ix, ix)]
                ba, aa = bn[ix], an[ix]
                c = np.linalg.lstsq(h, ba, rcond=1e-12)[0]
                q = np.linalg.lstsq(h, aa, rcond=1e-12)[0]
                if (not np.allclose(h @ c, ba, rtol=tolerance, atol=tolerance)
                        or not np.allclose(h @ q, aa, rtol=tolerance, atol=tolerance)):
                    continue
                center_squared = float(ba @ c)
                tangent_squared = proposal_norm_squared - float(aa @ q)
            else:
                ix = np.empty(0, dtype=int)
                c = q = np.empty(0)
                center_squared = 0.0
                tangent_squared = proposal_norm_squared
            error_scale = max(1.0, proposal_norm_squared, radius * radius, abs(center_squared))
            if center_squared < -tolerance * error_scale or tangent_squared < -tolerance * error_scale:
                continue
            center_squared, tangent_squared = max(0., center_squared), max(0., tangent_squared)
            available = radius * radius - center_squared
            radius_squared_tolerance = tolerance * max(radius * radius, center_squared, np.finfo(float).tiny)
            if available < -radius_squared_tolerance:
                continue
            if tangent_squared:
                fraction = min(1.0, math.sqrt(max(0., available) / tangent_squared))
            else:
                fraction = 1.0
            wn = np.zeros(m)
            wn[ix] = c - fraction * q
            values = fraction * an + gn @ wn
            slack = values - bn
            if np.any(slack < -tolerance * np.maximum(1., np.abs(bn))):
                continue
            squared = (fraction * fraction * proposal_norm_squared
                       + 2 * fraction * float(wn @ an) + float(wn @ gn @ wn))
            distance_squared = ((fraction - 1) ** 2 * proposal_norm_squared
                                + 2 * (fraction - 1) * float(wn @ an)
                                + float(wn @ gn @ wn))
            if min(squared, distance_squared) < -tolerance * error_scale:
                continue
            squared, distance_squared = max(0., squared), max(0., distance_squared)
            if squared > radius * radius + tolerance * max(radius * radius, np.finfo(float).tiny):
                continue
            result = Projection(
                proposal_scale=fraction,
                row_weights=tuple(float(x) for x in wn / scales),
                predicted_constraint_values=tuple(float(x) for x in values * scales),
                delta_norm=math.sqrt(squared),
                distance_to_proposal=math.sqrt(distance_squared),
                active_constraints=active,
                ball_active=abs(math.sqrt(squared) - radius) <= tolerance * max(1., radius),
                minimum_slack=float(slack.min()) if m else math.inf,
            )
            if best is None or result.distance_to_proposal < best.distance_to_proposal:
                best = result
    return best


def from_dense(rows, proposal, bounds, radius):
    """Convenience for small CPU experiments; large models should stream G/a."""
    rows = np.asarray(rows, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    if rows.ndim != 2 or proposal.shape != (rows.shape[1],):
        raise ValueError('Dense direction dimensions disagree')
    result = project_gram(rows @ rows.T, rows @ proposal, bounds,
                          float(proposal @ proposal), radius)
    if result is None:
        return None, None
    delta = result.proposal_scale * proposal + rows.T @ np.asarray(result.row_weights)
    return delta, result
