"""
mc_bot.py  --  "simulate every move, then let Laya pick" (the proposed design).

This is a faithful build of the plan: for every legal move, run randomised
playouts to the end of the game and record the average score differential; keep
sampling until the compute budget runs out; then hand the per-move statistics to
Laya to choose. It exists so the design can be measured instead of argued about.

    bot = MCBot(time_budget=2.0, chooser="argmax")   # simulation decides
    bot = MCBot(time_budget=2.0, chooser="laya")     # Laya reads the stats

The two choosers share the identical simulation, so a match between them isolates
exactly one thing: whether passing the numbers through Laya helps or hurts.

ONE DEVIATION FROM THE PLAN, AND WHY
------------------------------------
The plan says "each cell". The unit here is (cell, letter), because a cell is not
a move: an O next to a lone S hands the opponent an immediate SOS, while an S on
that same cell is usually safe. Scoring a cell would average those two together
and lose the distinction the game turns on. So 64 cells become up to 128 moves.
"""

import random
import time

from sos_engine import NC, S, O, Position, action_to_move, pick_quiet


def playout(pos, rng, eps):
    """One randomised game to the end; differential for the side to move.

    With probability `eps` a uniformly random legal move replaces the policy
    move, which is what makes repeated samples informative rather than identical.
    The turn rule is applied from the result of the move itself, so a random move
    that happens to score still correctly keeps the turn.
    """
    stack = []
    diff = 0
    sign = 1
    empty, threats, tc = pos.empty, pos.threats, pos.tcount
    while empty:
        if rng.random() < eps:
            idx = empty[rng.randrange(len(empty))]
            mv = idx if rng.random() < 0.5 else NC + idx
        elif threats:
            mv = max(threats, key=tc.__getitem__)
        else:
            mv = pick_quiet(pos)
        gained, tok = pos.make(mv)
        stack.append((mv, tok))
        if gained:
            diff += sign * gained          # scored: keep the turn
        else:
            sign = -sign                   # quiet: hand over
    for mv, tok in reversed(stack):
        pos.unmake(mv, tok)
    return diff


class MCBot:
    def __init__(self, wrap_around=True, time_budget=2.0, chooser="argmax",
                 eps=0.15, seed=None, shortlist=8, model="typed-decisions",
                 verbose=False):
        self.wrap_around = wrap_around
        self.time_budget = time_budget
        self.chooser = chooser              # "argmax" | "laya" | "random"
        self.eps = eps
        self.shortlist = shortlist
        self.verbose = verbose
        self.pos = Position(wrap_around)
        self.rng = random.Random(seed)
        self.laya = None
        if chooser == "laya":
            from laya_bot import LayaBot
            self.laya = LayaBot(wrap_around=wrap_around, model=model, mode="none")
        self.last_samples = 0
        self.last_moves = 0
        self.last_stats = []
        self.last_choice = None
        self.disagreements = 0
        self.calls = 0

    def _legal(self):
        return [a for idx in self.pos.empty for a in (idx, NC + idx)]

    def choose_action(self, board, my_score=0, opp_score=0):
        pos = self.pos
        if pos.wrap != self.wrap_around:
            pos.set_topology(self.wrap_around)
        pos.set_from_chars(board)
        if not pos.empty:
            return 0

        moves = self._legal()
        total = [0] * len(moves)
        runs = [0] * len(moves)
        rng, eps = self.rng, self.eps
        deadline = time.perf_counter() + self.time_budget

        # Round-robin sweeps: every move gets the same number of samples, and we
        # stop mid-sweep when the clock runs out. "n iterations based on compute
        # time", exactly as proposed.
        samples = 0
        while time.perf_counter() < deadline:
            for i, mv in enumerate(moves):
                if time.perf_counter() >= deadline:
                    break
                gained, tok = pos.make(mv)
                # A scoring move keeps the turn, so the playout runs for us;
                # otherwise it runs for the opponent and its value is negated.
                v = playout(pos, rng, eps)
                pos.unmake(mv, tok)
                total[i] += (gained + v) if gained else -v
                runs[i] += 1
                samples += 1
        self.last_samples = samples
        self.last_moves = len(moves)

        stats = [(total[i] / runs[i], runs[i], moves[i])
                 for i in range(len(moves)) if runs[i]]
        if not stats:                       # budget too small for one sweep
            return moves[0]
        stats.sort(key=lambda t: -t[0])
        self.last_stats = stats
        best_action = stats[0][2]

        if self.chooser == "argmax":
            self.last_choice = best_action
            return best_action
        if self.chooser == "random":
            self.last_choice = stats[self.rng.randrange(min(self.shortlist,
                                                            len(stats)))][2]
            return self.last_choice

        # chooser == "laya": hand it the simulation's own numbers.
        top = stats[:self.shortlist]
        criteria, actions = {}, {}
        for mean, n, a in top:
            label = "%s-%s" % (_square(a & 63), "S" if a < NC else "O")
            criteria[label] = ("simulated score %+.1f over %d games" % (mean, n))
            actions[label] = a
        self.calls += 1
        answers = self.laya.ask_choice(self._state(board, my_score, opp_score),
                                       criteria)
        pick = answers if answers in actions else None
        chosen = actions[pick] if pick else best_action
        if chosen != best_action:
            self.disagreements += 1
        self.last_choice = chosen
        if self.verbose:
            print("[MC] sim best %s | laya %s | %d samples"
                  % (_square(best_action & 63), pick, samples))
        return chosen

    def _state(self, board, my_score, opp_score):
        rows = ["%d %s" % (8 - r, "".join(ch if ch != ' ' else '.' for ch in board[r]))
                for r in range(8)]
        rows.append("  ABCDEFGH")
        return ("You are playing SOS.\nBoard:\n%s\n\nYour score %d, opponent %d."
                % ("\n".join(rows), my_score, opp_score))

    def choose_move(self, board, my_score=0, opp_score=0):
        return action_to_move(self.choose_action(board, my_score, opp_score))

    def close(self):
        if self.laya is not None:
            self.laya.close()


def _square(idx):
    r, c = divmod(idx, 8)
    return "%s%d" % ("ABCDEFGH"[c], 8 - r)


if __name__ == "__main__":
    import sys
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    b = [[' '] * 8 for _ in range(8)]
    b[3][3] = 'S'; b[3][5] = 'S'; b[4][4] = 'O'; b[2][2] = 'S'
    bot = MCBot(time_budget=budget, chooser="argmax", seed=1)
    t0 = time.perf_counter()
    mv = bot.choose_move(b)
    print("move %s | %d samples over %d moves in %.1fs (%.0f samples/move)"
          % (mv, bot.last_samples, bot.last_moves, time.perf_counter() - t0,
             bot.last_samples / bot.last_moves))
    for mean, n, a in bot.last_stats[:5]:
        print("   %-6s mean %+.2f over %d" % (_square(a & 63), mean, n))
