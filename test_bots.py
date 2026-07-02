"""
test_bots.py — standalone validation for the SOS AI bots (no pytest needed).

Run:  python test_bots.py           # ~30s: correctness + legality + timing
      python test_bots.py match     # adds an 8-game SmartBot-vs-greedy match

Checks:
  1. SmartBot._count_sos agrees exactly with game_logic's scoring on thousands
     of random placements (toroidal and standard boards).
  2. SmartBot plays only legal moves through complete games in both modes.
  3. Move latency stays within the time budget on worst-case sparse boards.
  4. (optional) SmartBot convincingly beats the legacy greedy bot.
"""

import random
import sys
import time

import numpy as np

from game_logic import SOSGame
from greedy_bot import SOSBot as GreedyBot
from smart_bot import SmartBot

FAILS = 0


def check(cond, msg):
    global FAILS
    if not cond:
        FAILS += 1
        print(f"  FAIL: {msg}")


def char_board(int_board):
    return [[' ' if v == 0 else ('S' if v == 1 else 'O') for v in row]
            for row in int_board]


def test_scoring_equivalence(games_per_mode=20):
    print("[1] scoring equivalence vs game_logic")
    for wrap in (True, False):
        bot = SmartBot(wrap_around=wrap, time_budget=0.0)
        rng = random.Random(42 if wrap else 43)
        mismatches = placements = 0
        for _ in range(games_per_mode):
            game = SOSGame(board_size=8, wrap_around=wrap)
            game.reset()
            while not np.all(game.board != 0):
                action = rng.choice(game.get_valid_actions())
                idx, piece = action % 64, (1 if action < 64 else 2)
                flat = [int(v) for v in game.board.flatten()]
                predicted = bot._count_sos(flat, idx, piece)
                mover = game.current_player
                before = game.scores[mover]
                game.step(action)
                if predicted != game.scores[mover] - before:
                    mismatches += 1
                placements += 1
        check(mismatches == 0, f"wrap={wrap}: {mismatches} scoring mismatches")
        print(f"  wrap={wrap}: {placements} placements, {mismatches} mismatches")


def test_legality():
    print("[2] full-game legality (SmartBot self-play, both modes)")
    for wrap in (True, False):
        game = SOSGame(8, wrap)
        game.reset()
        bot = SmartBot(wrap_around=wrap, time_budget=0.2)
        moves = 0
        while not np.all(game.board != 0):
            (r, c), letter = bot.choose_move(char_board(game.board))
            check(letter in ('S', 'O'), f"bad letter {letter!r}")
            check(game.board[r, c] == 0, f"illegal move at {(r, c)}")
            game.step((r * 8 + c) + (0 if letter == 'S' else 64))
            moves += 1
        check(moves == 64, f"game ended after {moves} moves")
        print(f"  wrap={wrap}: 64/64 legal, final {dict(game.scores)}")


def test_timing(budget=9.0):
    print("[3] timing (must stay under budget + 1s)")
    bot = SmartBot(wrap_around=True, time_budget=budget)
    sparse = [(0, 0, 'S'), (2, 2, 'S'), (4, 4, 'O'), (6, 6, 'S'),
              (1, 5, 'O'), (5, 1, 'S'), (3, 7, 'O'), (7, 3, 'S')]
    for name, pieces in {"empty": [], "midgame-2": [(3, 3, 'S'), (3, 5, 'S')],
                         "sparse-8": sparse}.items():
        b = [[' '] * 8 for _ in range(8)]
        for r, c, ch in pieces:
            b[r][c] = ch
        t0 = time.perf_counter()
        bot.choose_move(b)
        dt = time.perf_counter() - t0
        check(dt < budget + 1.0, f"{name}: {dt:.2f}s over budget")
        print(f"  {name}: {dt:.2f}s")


def test_match(n_games=8):
    print(f"[4] match: SmartBot vs greedy ({n_games} games, alternating)")
    wins = [0, 0, 0]
    for i in range(n_games):
        game = SOSGame(8, True)
        game.reset()
        smart = SmartBot(True, 0.5)
        greedy = GreedyBot(wrap_around=True)
        bots = {i % 2: smart, 1 - i % 2: greedy}
        while not np.all(game.board != 0):
            (r, c), L = bots[game.current_player].choose_move(
                char_board(game.board))
            game.step((r * 8 + c) + (0 if L == 'S' else 64))
        me = game.scores[i % 2]
        op = game.scores[1 - i % 2]
        wins[0 if me > op else 1 if op > me else 2] += 1
        print(f"  g{i+1}: smart={me:2d} greedy={op:2d}")
    check(wins[0] > wins[1], f"SmartBot did not out-win greedy: {wins}")
    print(f"  totals: smart={wins[0]} greedy={wins[1]} draws={wins[2]}")


if __name__ == "__main__":
    test_scoring_equivalence()
    test_legality()
    test_timing()
    if "match" in sys.argv[1:]:
        test_match()
    print()
    print("RESULT:", "ALL PASS" if FAILS == 0 else f"{FAILS} FAILURES")
    sys.exit(1 if FAILS else 0)
