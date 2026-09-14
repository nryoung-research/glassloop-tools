"""QUILLLIB - the second novel mini-API (Cohort B-easy of the capability accumulation pilot).

Deliberately ORTHOGONAL to vexlib: it operates on STRINGS (vex operates on integer lists), its helper names share no vocabulary with
vex (chunk/spin/prune/braid/weave/tally vs fold/tilt/sift/knot/lace/ledger), and its twists are unrelated. Every function has a
DELIBERATE TWIST that is not guessable from its name, so a model that has not learned the API cannot pass by priors alone.
Semantics are frozen once the family file is hashed. This file is the ground truth used by the task harness at test time.
"""


def fold(s, k):
    """Split s into consecutive pieces of length k and join them with '|'.
    TWIST: the final short piece is padded with '_' up to length k.
    fold("abcde", 2) -> "ab|cd|e_"
    """
    if k <= 0:
        raise ValueError("k must be positive")
    parts = []
    for i in range(0, len(s), k):
        p = s[i:i + k]
        parts.append(p + "_" * (k - len(p)))
    return "|".join(parts)


def tilt(s, n):
    """Rotate s left by n characters.
    TWIST: after rotating, the character that moved into the LAST position is upper-cased.
    tilt("abcd", 1) -> "bcdA"
    """
    if not s:
        return s
    m = n % len(s)
    r = s[m:] + s[:m]
    return r[:-1] + r[-1].upper()


def sift(s, pred_name):
    """Remove characters for which the named predicate holds ('vowel', 'digit', 'upper').
    TWIST: the removed characters are returned APPENDED at the end in reverse order, after a '#'.
    sift("a1b2", "digit") -> "ab#21"
    """
    preds = {"vowel": lambda c: c.lower() in "aeiou", "digit": lambda c: c.isdigit(), "upper": lambda c: c.isupper()}
    p = preds[pred_name]
    kept = "".join(c for c in s if not p(c))
    gone = "".join(c for c in s if p(c))
    return kept + "#" + gone[::-1]


def knot(s):
    """Interleave the first half and the second half of s character by character.
    TWIST: when the length is odd, the middle character is dropped, not kept.
    knot("abcde") -> "adbe"
    """
    n = len(s) // 2
    a, b = s[:n], s[len(s) - n:]
    return "".join(x + y for x, y in zip(a, b))


def lace(s, t):
    """Append t to s.
    TWIST: t is inserted AFTER THE FIRST CHARACTER of s, not at the end; empty s returns t.
    lace("abc", "XY") -> "aXYbc"
    """
    if not s:
        return t
    return s[0] + t + s[1:]


def ledger(s):
    """Count characters into a dict.
    TWIST: only characters that occur an EVEN number of times are reported, and keys are upper-cased.
    ledger("aabbbc") -> {"A": 2}
    """
    c = {}
    for ch in s:
        c[ch] = c.get(ch, 0) + 1
    return {k.upper(): v for k, v in c.items() if v % 2 == 0}


SPEC = '''The quill library (exact semantics):
- fold(s, k): split s into consecutive pieces of length k joined by "|"; the final short piece is padded with "_" to length k.
- tilt(s, n): rotate s left by n; the character that lands in the last position is upper-cased.
- sift(s, pred): remove the characters satisfying pred ("vowel", "digit" or "upper"); append "#" and then the removed characters in reverse order.
- knot(s): interleave the first and second halves character by character; with odd length the middle character is dropped.
- lace(s, t): insert t immediately after the first character of s (empty s returns t).
- ledger(s): a dict of the characters occurring an even number of times, keys upper-cased, values the counts.'''
