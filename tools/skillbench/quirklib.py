"""quirklib -- an INVENTED micro-library for the skills-pilot experiment.

Every function below has a made-up name and deliberately arbitrary (but exact,
deterministic, pure) semantics. The semantics are canonical AS WRITTEN HERE:
do not "correct" them to something more natural. Domain: ints and lists of
ints only. `%` and `//` follow Python's floor semantics (e.g. -7 % 3 == 2).

Each function's docstring also declares a unique NICKNAME. Skill-tier task
instructions refer to functions ONLY by nickname, never by real name.

No external dependencies. All functions return new objects (inputs are never
mutated).
"""


def snerp(xs, k):
    """Nickname: "the third-turn scramble".

    Rotate the list LEFT by (k mod len(xs)) positions, then negate the
    elements at indices 0, 3, 6, ... (every index divisible by 3, 0-based)
    of the ROTATED list. Returns a new list. If xs is empty, returns [].

    Example: snerp([3, 1, 2], 1) == [-1, 2, 3]
             (rotate-left 1 -> [1, 2, 3]; negate index 0 -> [-1, 2, 3])
    """
    n = len(xs)
    if n == 0:
        return []
    j = k % n
    rot = list(xs[j:]) + list(xs[:j])
    return [-v if i % 3 == 0 else v for i, v in enumerate(rot)]


def gribble(xs):
    """Nickname: "the parity weave".

    Take the elements at even indices (0, 2, 4, ...) in order, then append
    the elements at odd indices (1, 3, 5, ...) in REVERSED order with 1
    subtracted from each. Returns a new list.

    Example: gribble([5, 2, 8, 1, 9]) == [5, 8, 9, 0, 1]
             (evens [5, 8, 9]; odds [2, 1] reversed [1, 2] minus one [0, 1])
    """
    evens = list(xs[0::2])
    odds = list(xs[1::2])
    return evens + [v - 1 for v in reversed(odds)]


def quangle(a, b):
    """Nickname: "the crooked product".

    If (a + b) is even, return a*b - (a + b); otherwise return a*b + (a - b).

    Example: quangle(3, 5) == 7   (3+5 even: 15 - 8)
             quangle(6, 2) == 4   (6+2 even: 12 - 8)
             quangle(4, 1) == 7   (4+1 odd:  4 + 3)
    """
    if (a + b) % 2 == 0:
        return a * b - (a + b)
    return a * b + (a - b)


def mervin(xs, k):
    """Nickname: "the sieve stamp".

    Requires k >= 1. Keep the elements xs[i] for which (xs[i] + i) % k == 0
    (Python floor mod, i is the 0-based index), preserving order, then append
    ONE extra element: the count of elements that were dropped. Returns a new
    list (always non-empty: at minimum it is [len(xs)]).

    Example: mervin([4, 2, 7, 1], 2) == [4, 1, 2]
             (keep 4 (4+0 even) and 1 (1+3 even); dropped 2 -> append 2)
    """
    kept = [v for i, v in enumerate(xs) if (v + i) % k == 0]
    return kept + [len(xs) - len(kept)]


def drossel(xs):
    """Nickname: "the seesaw collapse".

    Return (sum of elements at even indices) minus TWICE (sum of elements at
    odd indices). Empty list gives 0.

    Example: drossel([5, 1, 4, 2, 3]) == 6   ((5+4+3) - 2*(1+2))
    """
    return sum(xs[0::2]) - 2 * sum(xs[1::2])


def plonk(x, k):
    """Nickname: "the parity fork".

    If x is even (x % 2 == 0, so e.g. -4 counts as even), return x * k;
    otherwise return x + k*k.

    Example: plonk(4, 3) == 12
             plonk(5, 3) == 14
    """
    if x % 2 == 0:
        return x * k
    return x + k * k


def varnick(xs, y):
    """Nickname: "the ledger polish".

    Map each element v of xs to: 2*v - y if v >= y, else v + y.
    Returns a new list of the same length.

    Example: varnick([1, 5, 2], 3) == [4, 7, 5]
             (1<3 -> 1+3; 5>=3 -> 10-3; 2<3 -> 2+3)
    """
    return [2 * v - y if v >= y else v + y for v in xs]


def smelk(xs):
    """Nickname: "the mirror climb".

    Reverse the list, then add each element's NEW 0-based index to it.
    Returns a new list of the same length.

    Example: smelk([4, 9, 2]) == [2, 10, 6]
             (reversed [2, 9, 4]; +0, +1, +2)
    """
    return [v + i for i, v in enumerate(reversed(xs))]


def torvel(xs, k):
    """Nickname: "the wheel tally".

    Requires k >= 1. Return the sum of (v % k) over all elements v (Python
    floor mod, so results are in 0..k-1 even for negative v), minus len(xs).

    Example: torvel([7, 3, 8], 3) == 0   ((1 + 0 + 2) - 3)
    """
    return sum(v % k for v in xs) - len(xs)


def brumble(x):
    """Nickname: "the gnarl step".

    If x is even, return x // 2 - 3 (Python floor division); if x is odd,
    return 3*x - 5.

    Example: brumble(10) == 2
             brumble(7) == 16
             brumble(-4) == -5   (-4 // 2 == -2)
    """
    if x % 2 == 0:
        return x // 2 - 3
    return 3 * x - 5
