"""
strong_bot.py  --  alpha-beta search bot for SOS. Drop-in for greedy/smart bot:

    bot = StrongBot(wrap_around=True, time_budget=2.0)
    (r, c), letter = bot.choose_move(board)     # board: list[list] of ' '/'S'/'O'

WHAT THE SEARCH ACTUALLY DOES
-----------------------------
Negamax over score differential. The one twist in SOS is the bonus turn: a
scoring move does NOT pass the turn, so

    value(move) = points + value(child, same side to move)      # scored
    value(move) =        - value(child, other side to move)     # quiet

Alpha/beta pass straight through on the scoring branch (still the same
maximiser) and are negated/swapped on the quiet branch. Because the accumulated
score is only a constant offset, value depends on the *board alone*, so the
transposition table is keyed on the board hash and is shared by both players,
by both move orders, and across moves of the real game.

DEPTH IS COUNTED IN TURNS, NOT PLACEMENTS
-----------------------------------------
A quiet move costs PLY (=4); a scoring move costs CPLY (=1). So depth measures
*hand-overs*, which is the unit that matters here, and the search follows a
cascade to its end almost for free instead of burning its whole budget on one
player's 20-point chain. A four-move cascade still costs one full turn of depth,
which bounds the tree.

QUIESCENCE
----------
Stopping at a fixed depth would let the search grab points and push the payback
just past the horizon. At depth 0 it therefore keeps searching *scoring moves
only*, with stand-pat 0 ("stop harvesting; assume the rest is even"). That is
cheap -- threats are few -- and it means a quiet move that hands the opponent a
30-point chain is scored as -30, not 0.

MOVE GENERATION
---------------
The killer optimisation. Position.touch marks cells lying on a live, already
started line. Cold cells (touch==0) are all equivalent waiting moves, so the
search expands ONE of them instead of ~50 identical ones. Hot cells are
classified by Position.quiet_features into safe (hands over nothing) and unsafe
(hands over a threat), and only a capped, diverse subset is expanded -- except
in the endgame, where every move is generated and the search is exact.
"""

import random
from time import perf_counter

from sos_engine import (NC, NACT, EMPTY, S, O, Position, action_to_move,
                        rollout)

INF = 1 << 20
PLY = 4          # depth cost of a quiet (turn-passing) move
CPLY = 1         # depth cost of a scoring move (bonus turn -> nearly free)


class TimeUp(Exception):
    """Raised inside the search when the move deadline passes."""


class StrongBot:
    def __init__(self, wrap_around=True, time_budget=2.0, exact_limit=12,
                 seed=None, tt_limit=1_200_000, leaf='rollout', width=100):
        self.leaf = leaf                    # 'rollout' | 'quiesce'
        self.width = width                  # percent scale on move-gen caps
        self.wrap_around = wrap_around
        self.time_budget = time_budget
        self.exact_limit = exact_limit      # <= this many empties -> full width
        self.tt_limit = tt_limit
        self.pos = Position(wrap_around)
        self.tt = {}
        # The playout is deterministic given the board, so its value is a pure
        # function of the position -- memoising it is exact, not an
        # approximation. This pays twice: transpositions inside one search, and
        # every shallower node re-visited by the next iterative-deepening pass.
        self.rt = {}
        self.hist = [0] * NACT
        self._rng = random.Random(seed)
        self.nodes = 0
        self.deadline = 0.0
        self._partial = None            # best root move of an aborted iteration
        # stats for the caller / debugging
        self.last_depth = 0
        self.last_nodes = 0
        self.last_value = 0
        self.last_solved = False
        # Root moves with their searched values, from the deepest COMPLETED
        # iteration. Exposed so another bot can pick among moves the search
        # rates equally without losing the search's strength -- see rank_moves.
        self.last_results = []

    # -- move generation -----------------------------------------------------
    def _caps(self, ne, in_chain):
        """(max safe, max unsafe) quiet candidates, or (None, None) for exact."""
        if ne <= self.exact_limit:
            return None, None
        if ne > 40:
            ks, ku = 6, 1
        elif ne > 24:
            ks, ku = 8, 2
        else:
            ks, ku = 14, 4
        if in_chain:
            # Mid-cascade the real question is only "keep going or stop", so a
            # couple of representative hand-over moves is enough.
            ks, ku = max(3, ks // 2), max(1, ku // 2)
        w = self.width
        if w != 100:
            ks, ku = max(2, ks * w // 100), max(1, ku * w // 100)
        return ks, ku

    def _gen(self, in_chain, tt_move, shuffle=False):
        pos = self.pos
        tc, touch, board, hist = pos.tcount, pos.touch, pos.board, self.hist
        empty = pos.empty
        ne = len(empty)
        ks, ku = self._caps(ne, in_chain)
        exact = ks is None

        scoring = sorted(pos.threats, key=lambda a: (-tc[a], -hist[a]))

        safe, unsafe, cold = [], [], -1
        qf = pos.quiet_features
        for idx in empty:
            if not touch[idx]:
                if exact:
                    safe.append((0, 0, -hist[idx], idx))
                    safe.append((0, 0, -hist[NC + idx], NC + idx))
                elif cold < 0:
                    cold = idx
                continue
            for base, letter in ((0, S), (NC, O)):
                a = base + idx
                if tc[a]:
                    continue                      # scoring move, already listed
                cr, ki, bu = qf(idx, letter)
                if cr:
                    unsafe.append((cr, -hist[a], a))
                else:
                    safe.append((-ki, -bu, -hist[a], a))

        if exact:
            safe.sort()
            unsafe.sort()
            quiet = [t[-1] for t in safe] + [t[-1] for t in unsafe]
        else:
            safe.sort()                                   # active: kill / build most
            picked = [t[-1] for t in safe[:ks - ks // 2]]
            seen = set(picked)
            for t in sorted(safe, key=lambda t: (-t[1], -t[0])):   # passive: quietest
                if len(picked) >= ks:
                    break
                if t[-1] not in seen:
                    seen.add(t[-1])
                    picked.append(t[-1])
            if cold >= 0:                                 # one pure waiting move
                picked.append(cold)
                picked.append(NC + cold)
            unsafe.sort()
            quiet = picked + [t[-1] for t in unsafe[:ku]]

        if shuffle:
            rng = self._rng
            rng.shuffle(quiet)
            # keep equal-scoring tactical moves in a random order too
            rng.shuffle(scoring)
            scoring.sort(key=lambda a: -tc[a])

        moves = scoring + quiet
        if tt_move >= 0 and board[tt_move & 63] == EMPTY:
            if moves and moves[0] == tt_move:
                return moves
            try:
                moves.remove(tt_move)
            except ValueError:
                pass
            moves.insert(0, tt_move)
        return moves

    # -- quiescence: harvest-only search -------------------------------------
    def _quiesce(self, alpha, beta):
        pos = self.pos
        threats = pos.threats
        if not threats:
            return 0
        best = 0                       # stand pat: stop, assume the rest is even
        if best >= beta:
            return best
        if best > alpha:
            alpha = best
        self.nodes += 1
        if not (self.nodes & 63) and perf_counter() > self.deadline:
            raise TimeUp
        tc = pos.tcount
        for mv in sorted(threats, key=lambda a: -tc[a]):
            g, tok = pos.make(mv)
            val = g + self._quiesce(alpha - g, beta - g)
            pos.unmake(mv, tok)
            if val > best:
                best = val
                if val > alpha:
                    alpha = val
                    if alpha >= beta:
                        break
        return best

    # -- main search ---------------------------------------------------------
    def _nega(self, depth, alpha, beta, in_chain):
        pos = self.pos
        if not pos.empty:
            return 0
        self.nodes += 1
        if not (self.nodes & 63) and perf_counter() > self.deadline:
            raise TimeUp
        if depth <= 0:
            if self.leaf != 'rollout':
                return self._quiesce(alpha, beta)
            rt = self.rt
            key = pos.hash
            v = rt.get(key)
            if v is None:
                v = rollout(pos)
                if len(rt) < self.tt_limit:
                    rt[key] = v
            return v

        tt = self.tt
        key = pos.hash
        entry = tt.get(key)
        tt_move = -1
        if entry is not None:
            d, v, flag, tt_move = entry
            if d >= depth:
                if flag == 0:
                    return v
                if flag == 1:
                    if v >= beta:
                        return v
                elif v <= alpha:
                    return v

        a0 = alpha
        best = -INF
        best_mv = -1
        for mv in self._gen(in_chain, tt_move):
            g, tok = pos.make(mv)
            if g:
                val = g + self._nega(depth - CPLY, alpha - g, beta - g, True)
            else:
                val = -self._nega(depth - PLY, -beta, -alpha, False)
            pos.unmake(mv, tok)
            if val > best:
                best = val
                best_mv = mv
                if val > alpha:
                    alpha = val
                    if alpha >= beta:
                        self.hist[mv] += depth * depth
                        break
        if best == -INF:                      # nothing generated: fall back
            return self._quiesce(a0, beta)
        if len(tt) < self.tt_limit:
            tt[key] = (depth, best,
                       0 if a0 < best < beta else (1 if best >= beta else 2),
                       best_mv)
        return best

    def _root(self, depth, moves):
        pos = self.pos
        alpha = -INF
        best_val, best_mv = -INF, moves[0]
        results = []
        for mv in moves:
            g, tok = pos.make(mv)
            if g:
                val = g + self._nega(depth - CPLY, alpha - g, INF, True)
            else:
                val = -self._nega(depth - PLY, -INF, -alpha, False)
            pos.unmake(mv, tok)
            results.append((val, mv))
            self._iter_results = results
            if val > best_val:
                best_val, best_mv = val, mv
                alpha = val
                self._partial = (val, mv)
        return best_val, best_mv, results

    # -- public API ----------------------------------------------------------
    def choose_action(self, board):
        pos = self.pos
        if pos.wrap != self.wrap_around:
            pos.set_topology(self.wrap_around)
            self.tt.clear()
            self.rt.clear()
        pos.set_from_chars(board)
        ne = len(pos.empty)
        self.last_results = []
        if ne == 0:
            return 0

        self.deadline = perf_counter() + self.time_budget
        self.nodes = 0
        self.hist = [0] * NACT
        self.last_solved = False
        self.last_depth = 0
        if len(self.tt) > self.tt_limit:
            self.tt.clear()
        if len(self.rt) > self.tt_limit:
            self.rt.clear()

        # Empty board: every cell is equivalent under the torus symmetry (and
        # near-equivalent on the bounded board), so searching buys nothing.
        if ne == NC:
            self.last_depth = 0
            self.last_value = 0
            opening = self._rng.randrange(NC)
            # Callers that read last_results (laya_bot) must still get a move
            # here: an empty board short-circuits the search, but it is not a
            # "no legal moves" position.
            self.last_results = [(0, opening)]
            return opening

        root_moves = self._gen(False, -1, shuffle=True)
        self.last_results = [(0, root_moves[0])]
        if len(root_moves) == 1:
            self.last_depth = 0
            return root_moves[0]

        best_val, best_mv = 0, root_moves[0]
        self._partial = None
        limit = ne * PLY
        depth = PLY
        while True:
            self._partial = None
            try:
                val, mv, results = self._root(depth, root_moves)
            except TimeUp:
                # Salvage a partial iteration only if it already beat the value
                # the previous, fully completed depth settled on.
                if self._partial is not None and self._partial[0] > best_val:
                    best_val, best_mv = self._partial
                break
            best_val, best_mv = val, mv
            self.last_depth = depth // PLY
            results.sort(key=lambda vm: -vm[0])
            self.last_results = list(results)
            root_moves = [m for _v, m in results]
            if depth >= limit:
                self.last_solved = True
                break
            if perf_counter() > self.deadline:
                break
            depth += PLY

        self.last_nodes = self.nodes
        self.last_value = best_val
        return best_mv

    def choose_move(self, board):
        return action_to_move(self.choose_action(board))

    def rank_moves(self, board):
        """[(value, action), ...] for the root moves, best first.

        Values are score differentials from the deepest completed iteration, so
        two moves with the same value are ones this search cannot separate. A
        caller may choose freely among those without giving anything up, which
        is what laya_bot.py does.
        """
        self.choose_action(board)
        return list(self.last_results)


# Drop-in aliases so `from strong_bot import SOSBot` swaps for the old bots.
SOSBot = StrongBot
SmartBot = StrongBot


if __name__ == "__main__":
    import sys

    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    b = [[' '] * 8 for _ in range(8)]
    for (r, c), ch in {(3, 3): 'S', (3, 5): 'S', (4, 4): 'O', (2, 2): 'S',
                       (5, 5): 'O', (1, 6): 'S', (6, 1): 'O'}.items():
        b[r][c] = ch
    bot = StrongBot(True, budget, seed=1)
    t0 = perf_counter()
    mv = bot.choose_move(b)
    print(f"move={mv}  depth={bot.last_depth}  nodes={bot.last_nodes:,}  "
          f"value={bot.last_value}  solved={bot.last_solved}  "
          f"time={perf_counter()-t0:.2f}s  "
          f"({bot.last_nodes/max(1e-9, perf_counter()-t0):,.0f} nodes/s)")
