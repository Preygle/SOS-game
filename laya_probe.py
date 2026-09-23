"""
laya_probe.py  --  does Laya actually pick the good move?

Builds positions where one shortlisted option is unambiguously best (it scores
several points for free while every alternative scores nothing), then asks each
checkpoint to choose. A model that understands the options should be near 100%.
Chance is 1/len(shortlist).

    python laya_probe.py                       # all three checkpoints
    python laya_probe.py multilingual --n 30
"""

import argparse
import json
import random
import subprocess
import sys

from laya_bot import LAYA_PYTHON, WORKER, INSTRUCTIONS, RULES, WRAP_NOTE, LayaBot


def positions(n, seed=4):
    """Random mid-game boards where a scoring move exists on the shortlist."""
    rng = random.Random(seed)
    out = []
    probe = LayaBot(wrap_around=True, policy="first")     # engine only, no model
    while len(out) < n:
        board = [[' '] * 8 for _ in range(8)]
        for cell in rng.sample(range(64), rng.randint(10, 34)):
            board[cell // 8][cell % 8] = rng.choice('SO')
        probe.pos.set_from_chars(board)
        if not probe.pos.empty:
            continue
        criteria, actions = probe._candidates()
        best, best_chain = None, 0
        for label, desc in criteria.items():
            chain = int(desc.split(", ")[1].split()[0]) if "with the chain" in desc else 0
            if chain > best_chain:
                best, best_chain = label, chain
        # keep only clear-cut cases: a real chain, and a shortlist worth choosing from
        if best and best_chain >= 2 and len(criteria) >= 4:
            out.append((board, criteria, best, best_chain))
    return out


def ask(model, cases, with_rules=True):
    proc = subprocess.Popen([LAYA_PYTHON, "-u", WORKER, "--model", model],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
    json.loads(proc.stdout.readline())
    hits, confs, ms, chance = 0, [], [], []
    for board, criteria, best, _chain in cases:
        grid = "\n".join("%d %s" % (8 - r, " ".join(ch if ch != ' ' else '.' for ch in board[r]))
                         for r in range(8)) + "\n  A B C D E F G H"
        state = (RULES.format(wrap=WRAP_NOTE) + "\n\nCURRENT GAME.\nBoard:\n" + grid) \
            if with_rules else ("You are playing SOS. Board:\n" + grid)
        proc.stdin.write(json.dumps({"state": state, "questions": {
            "move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}}) + "\n")
        proc.stdin.flush()
        rep = json.loads(proc.stdout.readline())
        if not rep.get("ok"):
            print("  error:", rep.get("error")[:120]); continue
        a = rep["answers"]["move"]
        hits += (a["choice"] == best)
        confs.append(a["confidence"])
        ms.append(rep["ms"])
        chance.append(1.0 / len(criteria))
    proc.stdin.write('{"cmd": "quit"}\n'); proc.stdin.flush(); proc.wait(timeout=10)
    n = len(ms)
    return hits / n, sum(confs) / n, sum(ms) / n, sum(chance) / n, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", default=["multilingual", "english", "typed-decisions"])
    p.add_argument("--n", type=int, default=25)
    args = p.parse_args()

    cases = positions(args.n)
    print(f"\n{len(cases)} positions where one shortlisted move scores a free chain "
          f"and the rest score nothing.\n")
    print(f"  {'checkpoint':<22} {'picks best':>11} {'chance':>7} {'confidence':>11} {'ms':>7}")
    print("  " + "-" * 62)
    for model in args.models:
        for with_rules in (True, False):
            acc, conf, ms, chance, n = ask(model, cases, with_rules)
            tag = f"{model}{'' if with_rules else ' (no rules)'}"
            print(f"  {tag:<22} {acc:>10.0%} {chance:>7.0%} {conf:>11.2f} {ms:>7.0f}")
    print()


if __name__ == "__main__":
    sys.exit(main())
