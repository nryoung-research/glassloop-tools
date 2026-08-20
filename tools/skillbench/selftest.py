"""Self-test: proves the scorer cannot lie in either direction.

For 10 sampled tasks (5 TIER-F + 5 TIER-S, drawn across train and holdout):

  (a) A programmatically generated CORRECT reference solution wrapped in
      THREE response formats (fenced-python, prose + bare fence, messy
      no-fence prose) must score 100% via every extraction path.
  (b) A plausible WRONG solution must score 0%:
        TIER-F wrong = an import-cheat (calls quirklib instead of recalling;
                       the tier's import block must defeat it),
        TIER-S wrong = the reference pipeline with a subtle extra step
                       (guaranteed to change every output).
      A non-parsing GARBAGE response must also score 0%.
      Both must log a failure reason and never crash the scorer.

Prints PASS/FAIL per case and exits nonzero on any failure.
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_tasks   # noqa: E402
import score_exec  # noqa: E402

GARBAGE = (
    "I'm honestly not sure this quirklib thing exists. My best guess is that\n"
    "it rotates something? Sorry.\n"
    "``` \n"
    "not even code here\n"
    "```\n"
)


def wrap_fenced(src):
    return "```python\n" + src + "\n```"


def wrap_prose(src):
    return ("Sure -- here is the solution you asked for.\n\n"
            "I worked through the nicknames first, then wrote it up:\n\n"
            "```\n" + src + "\n```\n\n"
            "Let me know if any test fails!")


def wrap_messy(src):
    return ("Okay, thinking out loud: the steps map onto quirklib calls "
            "I already know, so...\n\n" + src +
            "\n\nThat should do it (fingers crossed).")


def make_wrong(task, sol):
    if task["tier"] == "F":
        # Import-cheat: would be CORRECT if the import block failed.
        src = ("import quirklib\n\n"
               "def recall(case):\n"
               "    calls = {!r}\n"
               "    return getattr(quirklib, {!r})(*calls[case])\n"
               ).format(sol["cases"], sol["fn"])
    else:
        ref = sol["reference_solution"]
        inj = ("    v = v + [7]\n" if sol["returns"] == "list"
               else "    v = v + 7\n")
        assert "    return v" in ref
        src = ref.replace("    return v", inj + "    return v")
    return wrap_fenced(src)


def main():
    suite = gen_tasks.generate_suite()
    sols = suite["solutions"]
    pool_f = suite["train_F"] + suite["holdout_F"]
    pool_s = suite["train_S"] + suite["holdout_S"]
    rng = random.Random(7)
    tasks = rng.sample(pool_f, 5) + rng.sample(pool_s, 5)

    fails = 0
    total = 0
    for task in tasks:
        sol = sols[task["id"]]
        ref = sol["reference_solution"]
        cases = [
            ("fenced",  wrap_fenced(ref),  1.0),
            ("prose",   wrap_prose(ref),   1.0),
            ("messy",   wrap_messy(ref),   1.0),
            ("wrong",   make_wrong(task, sol), 0.0),
            ("garbage", GARBAGE,           0.0),
        ]
        for name, resp, want in cases:
            total += 1
            try:
                item = score_exec.score_response(task, resp, timeout=15.0)
            except Exception as e:  # scorer must NEVER crash on a response
                print("[FAIL] {:<14} {:<8} scorer crashed: {!r}".format(
                    task["id"], name, e))
                fails += 1
                continue
            ok = (item["score"] == want)
            if want == 0.0:
                ok = ok and bool(item["error_summary"])
            status = "PASS" if ok else "FAIL"
            detail = "score={:.2f} path={}".format(
                item["score"], item["extraction"]["path"])
            if want == 0.0:
                detail += " reason={!r}".format(
                    (item["error_summary"] or "")[:72])
            print("[{}] {:<14} {:<8} {}".format(
                status, task["id"], name, detail))
            if not ok:
                fails += 1
                print(json.dumps(item, indent=2, sort_keys=True)[:2500])

    print("-" * 60)
    print("selftest: {}/{} cases passed{}".format(
        total - fails, total, "" if fails == 0 else
        "  ({} FAILED)".format(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
