"""VEXLIB - the novel mini-API for CLIMBER rung 1.

A small, precisely specified sequence-transform library that no model can
have seen. Every function has a DELIBERATE TWIST that is not guessable
from its name, so a model that has not learned the API cannot pass by
priors alone; a model that HAS learned it can compose the functions on
novel inputs. That gap is exactly the capability rung 1 measures.

Semantics are frozen. This file is the ground truth used by the task
harness at test time; solutions must CALL these functions correctly.
"""


def chunk(xs, k):
    """Split xs into consecutive chunks of length k.
    TWIST: the final short chunk is padded with None up to length k.
    chunk([1,2,3,4,5], 2) -> [[1,2],[3,4],[5,None]]
    """
    if k <= 0:
        raise ValueError("k must be positive")
    out = []
    for i in range(0, len(xs), k):
        piece = list(xs[i:i + k])
        while len(piece) < k:
            piece.append(None)
        out.append(piece)
    return out


def weave(a, b):
    """Alternate elements of a and b, starting with a.
    TWIST: the remainder of the LONGER sequence is appended in REVERSE.
    weave([1,2],[9,8,7,6]) -> [1,9,2,8,6,7]
    """
    out = []
    n = min(len(a), len(b))
    for i in range(n):
        out.append(a[i])
        out.append(b[i])
    rest = list(a[n:]) if len(a) > len(b) else list(b[n:])
    out.extend(reversed(rest))
    return out


def tally(xs):
    """Count occurrences of each element.
    TWIST: only elements occurring TWO OR MORE times appear in the result.
    tally([1,1,2,3,3,3]) -> {1: 2, 3: 3}
    """
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return {k: v for k, v in counts.items() if v >= 2}


def spin(xs, n):
    """Rotate xs left by n (negative n rotates right).
    TWIST: the shift is reduced modulo len(xs)+1, NOT len(xs); if the
    reduced shift equals len(xs) it becomes 0. So spin(xs, len(xs)+1) is a
    no-op where an ordinary rotation would shift by one.
    spin([1,2,3,4], 1)  -> [2,3,4,1]
    spin([1,2,3,4], 5)  -> [1,2,3,4]   (5 % 5 == 0; ordinary would give
                                        [2,3,4,1])
    spin([1,2,3,4], 6)  -> [2,3,4,1]   (6 % 5 == 1; ordinary would give
                                        [3,4,1,2])
    spin([1,2,3,4], -1) -> [4,1,2,3]
    """
    if not xs:
        return []
    L = len(xs)
    m = abs(n) % (L + 1)
    if m == L:
        m = 0
    if n < 0:
        m = (L - m) % L
    return list(xs[m:]) + list(xs[:m])


def prune(xs, pred):
    """Remove every element for which pred(x) is True.
    TWIST: the FIRST matching element is KEPT (only later matches drop).
    prune([1,2,3,4], lambda x: x % 2 == 0) -> [1,2,3]
    """
    out = []
    seen_match = False
    for x in xs:
        if pred(x):
            if not seen_match:
                seen_match = True
                out.append(x)
            continue
        out.append(x)
    return out


def braid(xs):
    """Pair each element with the element that many positions ahead,
    wrapping around; unpaired positions map to themselves.
    TWIST: indices are 1-based for the offset computation.
    braid([10,20,30]) -> [(10,20),(20,10),(30,30)]
      i=0 (1-based 1): 10 pairs with xs[(0+1) % 3] = 20
      i=1 (1-based 2): 20 pairs with xs[(1+2) % 3] = 10
      i=2 (1-based 3): 30 pairs with xs[(2+3) % 3] = 30
    """
    n = len(xs)
    if n == 0:
        return []
    return [(xs[i], xs[(i + i + 1) % n]) for i in range(n)]


API = {"chunk": chunk, "weave": weave, "tally": tally,
       "spin": spin, "prune": prune, "braid": braid}

SPEC = """vex library reference:
  vex.chunk(xs, k)   -> consecutive k-length chunks; the final short chunk
                        is PADDED WITH None up to length k.
  vex.weave(a, b)    -> alternates a and b starting with a; the leftover
                        tail of the longer sequence is appended REVERSED.
  vex.tally(xs)      -> dict of element -> count, INCLUDING ONLY elements
                        that occur two or more times.
  vex.spin(xs, n)    -> rotate left by n (negative rotates right); the
                        shift is reduced modulo len(xs)+1, and a reduced
                        shift equal to len(xs) becomes 0.
  vex.prune(xs, pred)-> drop every element satisfying pred, EXCEPT the
                        first matching element, which is kept.
  vex.braid(xs)      -> for index i (0-based), pairs xs[i] with
                        xs[(2*i + 1) % len(xs)].
"""
