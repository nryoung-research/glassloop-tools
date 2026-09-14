"""Small mathematical reference only. No model loading or training integration."""

from dataclasses import dataclass
from itertools import combinations
import math
import struct


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(x):
    return math.sqrt(dot(x, x))


def solve_square(matrix, rhs, tol=1e-12):
    """Partial-pivot Gaussian elimination; return None for singular systems."""
    n = len(rhs)
    aug = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) <= tol:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col])]
    return [row[-1] for row in aug]


@dataclass(frozen=True)
class Direction:
    delta: tuple
    length: float
    active: tuple


def minnorm_direction(rows, bounds, trust_radius, tol=1e-9):
    """Solve min ||d|| with A d >= b, ||d|| <= radius, in <=6 dimensions.

    Enumerates small active sets and checks KKT conditions. This exponential
    reference is deliberately capped; it is NOT a large-model solver.
    """
    if not rows or len(rows) != len(bounds):
        raise ValueError("Require nonempty equally sized rows and bounds")
    n, m = len(rows[0]), len(rows)
    if not 1 <= n <= 6 or m > 16:
        raise ValueError("Reference limit: 1..6 dimensions, at most 16 constraints")
    if any(len(row) != n for row in rows):
        raise ValueError("Ragged matrix")
    if trust_radius < 0 or not math.isfinite(trust_radius):
        raise ValueError("Trust radius must be finite and nonnegative")
    if not all(math.isfinite(x) for row in rows for x in row):
        raise ValueError("Non-finite coefficient")
    if not all(math.isfinite(x) for x in bounds):
        raise ValueError("Non-finite bound")

    def feasible(delta):
        return all(dot(row, delta) >= bound - tol for row, bound in zip(rows, bounds))

    zero = (0.0,) * n
    if feasible(zero):
        return Direction(zero, 0.0, ())
    best = None
    for count in range(1, min(n, m) + 1):
        for active in combinations(range(m), count):
            chosen = [rows[i] for i in active]
            gram = [[dot(a, b) for b in chosen] for a in chosen]
            multipliers = solve_square(gram, [bounds[i] for i in active])
            if multipliers is None or min(multipliers) < -tol:
                continue
            delta = tuple(sum(lam * row[j] for lam, row in zip(multipliers, chosen)) for j in range(n))
            length = norm(delta)
            if feasible(delta) and (best is None or length < best.length):
                best = Direction(delta, length, active)
    if best is None or best.length > trust_radius + tol:
        return None
    return best


def bfloat16_scalar(value):
    """Round one float32 scalar to BF16, ties to even; not a BF16 model simulator."""
    if not math.isfinite(value):
        raise ValueError("Finite inputs required")
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bits))[0]


@dataclass(frozen=True)
class Guard:
    passed: bool
    reason: str
    minimum_margin: float
    checked_positions: int


def greedy_trace_guard(logits_at_prefix, prompt, completion, eos="<eos>", required_margin=0.0):
    """Check the expected winner against EVERY rival at EVERY prefix, including EOS.

    The callback supplies ALL finite token selection scores after processing,
    or raw logits when no logits processors are enabled. This reference handles
    EOS-only termination and rejects all nonfinite values, including masks.
    A mathematical callback check does not certify a different real runtime.
    """
    if not completion or completion[-1] != eos or eos in completion[:-1]:
        raise ValueError("Completion must terminate with exactly one final EOS")
    if required_margin < 0 or not math.isfinite(required_margin):
        raise ValueError("Margin must be finite and nonnegative")
    prefix = tuple(prompt)
    minimum = math.inf
    for index, expected in enumerate(completion):
        logits = logits_at_prefix(prefix)
        if expected not in logits or len(logits) < 2:
            return Guard(False, "missing expected token or rival", -math.inf, index + 1)
        if not all(math.isfinite(value) for value in logits.values()):
            return Guard(False, "non-finite logit", -math.inf, index + 1)
        margin = logits[expected] - max(value for token, value in logits.items() if token != expected)
        minimum = min(minimum, margin)
        # A strict comparison avoids relying on unspecified tie-breaking.
        if margin <= required_margin:
            return Guard(False, "winner margin failed at position %d" % index, minimum, index + 1)
        prefix += (expected,)
    return Guard(True, "all decisions and EOS preserved", minimum, len(completion))


@dataclass(frozen=True)
class Trial:
    accepted: bool
    scale: float
    attempts: int
    reason: str


def backtrack_transaction(state, delta, accepts, trust_radius, max_backtracks=12):
    """Try finite proposals; commit accepted state, otherwise restore original bytes.

    State is a list of ordinary floats. Production transactions would also need
    optimizer moments, buffers, serialization and decoder caches in their scope.
    The callback must include checks against an immutable ROOT, not just state.
    """
    if len(state) != len(delta):
        raise ValueError("Direction shape differs from state")
    if trust_radius < 0 or not math.isfinite(trust_radius):
        raise ValueError("Invalid trust radius")
    if not isinstance(max_backtracks, int) or max_backtracks < 0:
        raise ValueError("Invalid backtrack count")
    if not all(math.isfinite(value) for value in list(state) + list(delta)):
        raise ValueError("Finite state and direction required")
    original = list(state)
    last_reason = "no proposal"
    try:
        for attempt in range(max_backtracks + 1):
            scale = 0.5 ** attempt
            step = [scale * value for value in delta]
            if norm(step) > trust_radius:
                last_reason = "trust radius exceeded"
                continue
            proposal = [value + shift for value, shift in zip(original, step)]
            if not all(math.isfinite(value) for value in proposal):
                last_reason = "non-finite proposal"
                continue
            state[:] = proposal
            accepted, last_reason = accepts(tuple(state))
            if accepted:
                return Trial(True, scale, attempt + 1, last_reason)
            state[:] = original
    except BaseException:
        state[:] = original
        raise
    state[:] = original
    return Trial(False, 0.0, max_backtracks + 1, last_reason)


def silu(x):
    return x / (1.0 + math.exp(-x))


def gated_mlp(state, x):
    """One gated neuron: down * SiLU(gate*x) * (up*x), down fixed at 1."""
    up, gate = state
    return silu(gate * x) * up * x


def gated_gradient(state, x):
    up, gate = state
    t = gate * x
    sigmoid = 1.0 / (1.0 + math.exp(-t))
    dsilu = sigmoid + t * sigmoid * (1.0 - sigmoid)
    return (silu(t) * x, up * x * x * dsilu)


def compensated_point(root, target):
    """Analytic two-input toy solution; no corresponding LLM inverse is promised."""
    old = gated_mlp(root, 1.0)
    gate = math.log(old / target)
    if abs(silu(gate)) < 1e-12:
        raise ValueError("Target reaches singular toy parameterization")
    return (old / silu(gate), gate)


def demo():
    import json
    root = (1.0, 1.0)
    old, new = gated_mlp(root, 1.0), gated_mlp(root, -1.0)
    old_grad, new_grad = gated_gradient(root, 1.0), gated_gradient(root, -1.0)
    direction = minnorm_direction([old_grad, tuple(-x for x in old_grad), new_grad], [0.0, 0.0, 0.02], 1.0)
    state = list(root)

    def finite_check(candidate):
        drift = abs(gated_mlp(candidate, 1.0) - old)
        return drift <= 1e-4, "root output drift %.9g" % drift

    full = tuple(a + b for a, b in zip(root, direction.delta))
    trial = backtrack_transaction(state, direction.delta, finite_check, 1.0)
    exact = compensated_point(root, 0.5)

    def logits(prefix):
        if prefix == ("legacy-prompt",):
            return {"ok": 2.0 - 1000 * abs(gated_mlp(exact, 1.0) - old), "wrong": 1.0, "other": 0.5, "<eos>": -1.0}
        return {"ok": 0.0, "wrong": -1.0, "other": 1.0, "<eos>": 2.0}

    guard = greedy_trace_guard(logits, ("legacy-prompt",), ("ok", "<eos>"))
    report = {
        "scope": "CPU mathematical toy; no evidence that Qwen passes its screens",
        "base_old_output": old,
        "base_new_output": new,
        "independent_scalar_key_projection_direction": [0.0, 0.0],
        "functional_linear_direction": list(direction.delta),
        "predicted_old_change": dot(old_grad, direction.delta),
        "full_step_actual_old_change": gated_mlp(full, 1.0) - old,
        "full_step_postcheck_passed": finite_check(full)[0],
        "backtrack_scale_accepted": trial.scale,
        "accepted_root_output_drift": abs(gated_mlp(state, 1.0) - old),
        "compensated_parameters": list(exact),
        "compensated_old_output": gated_mlp(exact, 1.0),
        "compensated_new_output": gated_mlp(exact, -1.0),
        "trace_including_eos_preserved": guard.passed,
        "positions_checked": guard.checked_positions,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    demo()
