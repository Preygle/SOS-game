"""
test_strong_bot.py  --  correctness tests for the search engine.

Run:  python test_strong_bot.py

The interesting one is test_endgame_is_optimal: it brute-forces small endgames
with a reference solver that shares no code with sos_engine (it counts SOS with
greedy_bot's own scanner) and checks that the search reports the same value AND
picks a move that preserves it. That catches sign errors, bonus-turn handling
mistakes, and transposition-table bugs, which are exactly the things that fail
silently in a game bot -- it keeps playing, just badly.
"""

import random
import sys

from greedy_bot import SOSBot as _Ref
from sos_engine import move_to_action
from strong_bot import StrongBot

WRAP = True
_ref = _Ref(wrap_around=WRAP)


def _key(board):
    return ''.join(''.join(row) for row in board)


def ref_solve(board, memo):
    """Optimal future score differential for the side to move.

    Independent of sos_engine: uses greedy_bot's SOS scanner and plain recursion.
    A scoring move keeps the turn (so its value adds), a quiet move passes it
    (so its value negates) -- the whole rule set of SOS in four lines.
    """
    k = _key(board)
    hit = memo.get(k)
    if hit is not None:
        return hit
    empties = [(r, c) for r in range(8) for c in range(8) if board[r][c] == ' ']
    if not empties:
        return 0
    best = None
    for r, c in empties:
        for letter in ('S', 'O'):
            board[r][c] = letter
            pts = _ref._count_sos(board, r, c)
            val = pts + ref_solve(board, memo) if pts else -ref_solve(board, memo)
            board[r][c] = ' '
            if best is None or val > best:
                best = val
    memo[k] = best
    return best


def random_endgame(rng, empties):
    """A board with `empties` blanks, reached by legal-ish random filling."""
    board = [[' '] * 8 for _ in range(8)]
    cells = list(range(64))
    rng.shuffle(cells)
    for cell in cells[:64 - empties]:
        board[cell // 8][cell % 8] = rng.choice(('S', 'O'))
    return board


def test_endgame_is_optimal(n=12, empties=6, seed=11):
    rng = random.Random(seed)
    bot = StrongBot(wrap_around=WRAP, time_budget=5.0, seed=1)
    checked = 0
    for _ in range(n):
        board = random_endgame(rng, empties)
        want = ref_solve([row[:] for row in board], {})

        (r, c), letter = bot.choose_move([row[:] for row in board])
        assert board[r][c] == ' ', "engine returned an occupied cell"
        assert bot.last_solved, "engine did not claim to solve a 6-empty endgame"
        assert bot.last_value == want, \
            f"value {bot.last_value} != optimal {want}"

        # And the move it plays must actually preserve that value.
        board[r][c] = letter
        pts = _ref._count_sos(board, r, c)
        got = pts + ref_solve(board, {}) if pts else -ref_solve(board, {})
        assert got == want, f"played move worth {got}, optimal is {want}"
        checked += 1
    print(f"  endgame optimality: {checked} random {empties}-empty positions "
          f"solved exactly  OK")


def test_takes_free_points():
    """A double SOS is on the board; the engine must take it, not a single."""
    board = [[' '] * 8 for _ in range(8)]
    # Row 2: S . S  -> an O at (2,3) scores one.
    board[2][2] = board[2][4] = 'S'
    # (4,4) is the middle of two separate S..S lines, so an O there scores two.
    board[4][3] = board[4][5] = 'S'
    board[3][3] = board[5][5] = 'S'
    bot = StrongBot(wrap_around=WRAP, time_budget=1.0, seed=1)
    (r, c), letter = bot.choose_move([row[:] for row in board])
    board[r][c] = letter
    assert _ref._count_sos(board, r, c) == 2, \
        f"took {(r, c)}{letter} for {_ref._count_sos(board, r, c)} pt, a 2-pointer existed"
    print("  tactics: takes the double SOS rather than a single       OK")


def test_rarely_gifts(threshold=0.25):
    """Sanity check on quiet play, sampled from real self-play positions.

    Random boards are useless here -- they almost always contain a live threat,
    so the earlier version of this test silently examined zero positions. These
    come from an actual game instead, and only positions where nothing scores
    for the mover and a safe move exists are judged.

    This is a rate check, not an assertion of perfection: handing over a small
    SOS on purpose is a real technique in this game (it buys the parity of the
    endgame), so a strong engine is expected to do it occasionally.
    """
    from game_logic import SOSGame
    import numpy as np

    bot = StrongBot(wrap_around=WRAP, time_budget=0.4, seed=3)
    game = SOSGame(8, WRAP)
    game.reset()
    judged = gifted = 0
    while not np.all(game.board != 0):
        board = [[' ' if v == 0 else ('S' if v == 1 else 'O') for v in row]
                 for row in game.board]
        quiet = not any(_probe(board, r, c, L)
                        for r in range(8) for c in range(8) for L in 'SO')
        (r, c), letter = bot.choose_move([row[:] for row in board])
        if quiet and _has_safe_move(board):
            judged += 1
            board[r][c] = letter
            if max((_probe(board, rr, cc, LL) for rr in range(8)
                    for cc in range(8) for LL in 'SO'), default=0):
                gifted += 1
            board[r][c] = ' '
        game.step(move_to_action((r, c), letter))
    assert judged >= 5, f"only {judged} quiet positions sampled; test is vacuous"
    rate = gifted / judged
    assert rate <= threshold,         f"handed over a free SOS in {gifted}/{judged} avoidable positions"
    print(f"  quiet play: gifted a free SOS in {gifted}/{judged} avoidable "
          f"positions  OK")


def _probe(board, r, c, letter):
    if board[r][c] != ' ':
        return 0
    board[r][c] = letter
    n = _ref._count_sos(board, r, c)
    board[r][c] = ' '
    return n


def _has_safe_move(board):
    for r in range(8):
        for c in range(8):
            if board[r][c] != ' ':
                continue
            for letter in 'SO':
                board[r][c] = letter
                risk = max((_probe(board, rr, cc, LL)
                            for rr in range(8) for cc in range(8)
                            for LL in 'SO' if board[rr][cc] == ' '), default=0)
                board[r][c] = ' '
                if risk == 0:
                    return True
    return False


def test_legal_and_prompt():
    """Every move legal, and the clock respected, over a whole game."""
    from game_logic import SOSGame
    import numpy as np
    from time import perf_counter

    budget = 0.5
    bot = StrongBot(wrap_around=WRAP, time_budget=budget, seed=2)
    game = SOSGame(8, WRAP)
    game.reset()
    worst = 0.0
    while not np.all(game.board != 0):
        snap = [[' ' if v == 0 else ('S' if v == 1 else 'O') for v in row]
                for row in game.board]
        t0 = perf_counter()
        (r, c), letter = bot.choose_move(snap)
        worst = max(worst, perf_counter() - t0)
        assert snap[r][c] == ' ', "illegal move"
        game.step(move_to_action((r, c), letter))
    assert worst < budget * 2.0 + 0.3, f"move took {worst:.2f}s on a {budget}s budget"
    print(f"  timing: full game legal, slowest move {worst:.2f}s "
          f"on a {budget}s budget  OK")


if __name__ == "__main__":
    print("strong_bot tests")
    test_takes_free_points()
    test_legal_and_prompt()
    test_rarely_gifts()
    test_endgame_is_optimal()
    print("all passed")
    sys.exit(0)
