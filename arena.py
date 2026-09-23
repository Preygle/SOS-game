"""
arena.py  --  head-to-head bot measurement, parallel over CPU cores.

    python arena.py strong:1.0 smart:1.0 --games 20
    python arena.py strong:2.0 greedy --games 40 --workers 12
    python arena.py strong:1.0 neural --games 20 --no-wrap

Bot specs:  strong:<sec>[:k=v,k=v]  smart:<sec>  greedy  neural  random
            laya:<checkpoint>[:policy=laya|first|random]
            mc:<sec>[:chooser=argmax|laya|random]   -- per-move Monte Carlo
e.g. strong:1.0:exact_limit=18,leaf=quiesce   laya:multilingual:policy=first
Games alternate who moves first, so the first-player edge cancels out.
"""

import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from game_logic import SOSGame


# One bot per (spec, wrap) per worker process. Laya costs ~30 s to load a
# checkpoint and milliseconds to decide, so building a fresh one per game would
# measure the loader rather than the player. Reuse is safe for the others too:
# StrongBot's table is keyed on the board, so it stays valid across games.
_BOT_CACHE = {}


def get_bot(spec, wrap):
    key = (spec, wrap)
    if key not in _BOT_CACHE:
        _BOT_CACHE[key] = make_bot(spec, wrap)
    return _BOT_CACHE[key]


def make_bot(spec, wrap):
    name, _, rest = spec.partition(':')
    arg, _, extra = rest.partition(':')
    name = name.lower()
    if name == 'strong':
        from strong_bot import StrongBot
        kw = {}
        for item in filter(None, extra.split(',')):
            k, _, v = item.partition('=')
            try:
                kw[k] = int(v)
            except ValueError:
                kw[k] = v
        return StrongBot(wrap_around=wrap, time_budget=float(arg or 1.0), **kw)
    if name == 'smart':
        from smart_bot import SmartBot
        return SmartBot(wrap_around=wrap, time_budget=float(arg or 1.0))
    if name == 'greedy':
        from greedy_bot import SOSBot
        return SOSBot(wrap_around=wrap)
    if name == 'neural':
        from neural_bot import NeuralBot
        b = NeuralBot(wrap_around=wrap)
        if not b.available:
            raise SystemExit("neural bot unavailable (no checkpoint / no torch)")
        return b
    if name == 'laya':
        from laya_bot import LayaBot
        kw = {}
        for item in filter(None, extra.split(',')):
            k, _, v = item.partition('=')
            for cast in (int, float):        # time_budget/margin are numbers
                try:
                    v = cast(v)
                    break
                except ValueError:
                    continue
            kw[k] = v
        return LayaBot(wrap_around=wrap, model=(arg or 'typed-decisions'), **kw)
    if name == 'mc':
        from mc_bot import MCBot
        kw = {}
        for item in filter(None, extra.split(',')):
            k, _, v = item.partition('=')
            for cast in (int, float):
                try:
                    v = cast(v)
                    break
                except ValueError:
                    continue
            kw[k] = v
        return MCBot(wrap_around=wrap, time_budget=float(arg or 1.0), **kw)
    if name == 'random':
        return RandomBot()
    raise SystemExit(f"unknown bot spec: {spec}")


class RandomBot:
    def choose_move(self, board):
        legal = [(r, c) for r in range(8) for c in range(8) if board[r][c] == ' ']
        return random.choice(legal), random.choice(['S', 'O'])


def _chars(bi):
    return [[' ' if v == 0 else ('S' if v == 1 else 'O') for v in row] for row in bi]


def _action(move):
    (r, c), letter = move
    return r * 8 + c + (0 if letter == 'S' else 64)


def play_one(job):
    """One game. `first` selects which bot is player 0."""
    spec_a, spec_b, wrap, first, seed = job
    random.seed(seed)
    a, b = get_bot(spec_a, wrap), get_bot(spec_b, wrap)
    for bot in (a, b):
        if hasattr(bot, '_rng'):
            bot._rng = random.Random(seed)
    game = SOSGame(8, wrap)
    game.reset()
    seats = {first: a, 1 - first: b}
    while not np.all(game.board != 0):
        move = seats[game.current_player].choose_move(_chars(game.board))
        game.step(_action(move))
    return game.scores[first], game.scores[1 - first]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('bot_a')
    p.add_argument('bot_b')
    p.add_argument('--games', type=int, default=20)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument('--no-wrap', action='store_true')
    p.add_argument('--seed', type=int, default=1000)
    args = p.parse_args()
    wrap = not args.no_wrap

    jobs = [(args.bot_a, args.bot_b, wrap, i % 2, args.seed + i)
            for i in range(args.games)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(play_one, jobs))

    w = l = d = 0
    sa = sb = 0
    seat = {0: [0, 0, 0], 1: [0, 0, 0]}       # per seat: wins, losses, draws
    for i, (pa, pb) in enumerate(results):
        sa += pa
        sb += pb
        outcome = 0 if pa > pb else (1 if pb > pa else 2)
        seat[i % 2][outcome] += 1
        if outcome == 0:
            w += 1
        elif outcome == 1:
            l += 1
        else:
            d += 1
    n = len(results)
    print(f"\n{args.bot_a}  vs  {args.bot_b}   ({n} games, wrap={wrap})")
    print(f"  record : {w}-{l}-{d}   ({100.0 * (w + 0.5 * d) / n:.1f}% score)")
    print(f"  points : {sa / n:.1f}  vs  {sb / n:.1f}   (diff {(sa - sb) / n:+.1f})")
    # Split by seat: a bot that is fine going first and hopeless going second
    # has a different problem from one that is simply weaker.
    for s_i, tag in ((0, 'moving first'), (1, 'moving second')):
        ww, ll, dd = seat[s_i]
        tot = ww + ll + dd
        if tot:
            print(f"  {tag:<14}: {ww}-{ll}-{dd}"
                  f"   ({100.0 * (ww + 0.5 * dd) / tot:.0f}%)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
