"""
sos_engine.py  --  fast, exact core for 8x8 SOS (toroidal or bounded).

WHY THIS FILE EXISTS
--------------------
The old bots were both effectively 1-ply:
  * smart_bot.py  = one deterministic greedy rollout per candidate move,
  * neural_bot.py = one network forward pass + a greedy tactical override.
Neither can see a forced sequence, which is exactly what decides SOS: because a
scoring move grants a bonus turn, the game is a cascade/zugzwang game (the same
shape as Dots-and-Boxes). Whoever is *forced* to open the board hands over a
chain worth 10-40 points, and that is invisible to a 1-ply search.

THE LINE REPRESENTATION
-----------------------
An SOS occupies three collinear cells (a, b, c) and is complete iff
board[a] is S, board[b] is O and board[c] is S. On the 8x8 torus there are
exactly 64 middles x 4 axes = 256 such *lines*, and every cell lies on 12 of
them (4 as the middle, 8 as an end). So the whole game is 256 three-slot
patterns.

For each line we keep
    filled[l] = how many of its slots already hold the letter that line needs
    dead[l]   = how many slots hold the wrong letter (a dead line can never score)
and maintain them incrementally: a placement touches exactly 12 lines.

That gives, for free and in O(1):
    * points for a move   -- tcount[action]
    * the full threat set -- lines with filled==2 and dead==0; such a line has
                             exactly one empty slot, so it names the (cell,
                             letter) that scores it.
This is what makes a real search affordable in pure Python.

VALUE FUNCTION / TRANSPOSITIONS
-------------------------------
Search returns the optimal *future* score differential for the side to move.
Accumulated scores are only a constant offset and the rules depend on nothing
but the letters on the board, so that value is a function of the board alone --
not of whose turn it is and not of the score. The transposition table can
therefore be keyed on the board hash only, and entries are shared by both
players and across move orders.
"""

import random

N = 8
NC = N * N                       # 64 cells
S, O, EMPTY = 1, 2, 0
AXES = ((0, 1), (1, 0), (1, 1), (1, -1))

# Action encoding matches game_logic: 0-63 place S, 64-127 place O.
NACT = 2 * NC

_TABLES = {}


def tables(wrap):
    """(LINES, CELL_LINES) for the requested topology, memoised.

    LINES[l]        = (a, b, c)  -- needs S at a, O at b, S at c
    CELL_LINES[idx] = ((line_id, required_letter), ...) for every line through idx
    """
    t = _TABLES.get(wrap)
    if t is not None:
        return t
    lines = []
    for br in range(N):
        for bc in range(N):
            for dr, dc in AXES:
                ar, ac, cr, cc = br - dr, bc - dc, br + dr, bc + dc
                if wrap:
                    a = (ar % N) * N + (ac % N)
                    c = (cr % N) * N + (cc % N)
                else:
                    if not (0 <= ar < N and 0 <= ac < N
                            and 0 <= cr < N and 0 <= cc < N):
                        continue
                    a, c = ar * N + ac, cr * N + cc
                lines.append((a, br * N + bc, c))
    cl = [[] for _ in range(NC)]
    for l, (a, b, c) in enumerate(lines):
        cl[a].append((l, S))
        cl[b].append((l, O))
        cl[c].append((l, S))
    t = (tuple(lines), tuple(tuple(x) for x in cl))
    _TABLES[wrap] = t
    return t


# Fixed Zobrist keys: reproducible across runs so the TT behaves identically.
_zr = random.Random(20260101)
ZOB = (tuple(0 for _ in range(NC)),
       tuple(_zr.getrandbits(62) for _ in range(NC)),
       tuple(_zr.getrandbits(62) for _ in range(NC)))


class Position:
    """Mutable board with make/unmake and incrementally maintained threats."""

    __slots__ = ('wrap', 'lines', 'cl', 'board', 'filled', 'dead',
                 'tcount', 'threats', 'empty', 'epos', 'hash', 'touch', 'hot')

    def __init__(self, wrap=True):
        self.set_topology(wrap)
        self.reset()

    def set_topology(self, wrap):
        self.wrap = wrap
        self.lines, self.cl = tables(wrap)

    def reset(self):
        self.board = [EMPTY] * NC
        self.filled = [0] * len(self.lines)
        self.dead = [0] * len(self.lines)
        self.tcount = [0] * NACT
        self.threats = set()
        self.empty = list(range(NC))
        self.epos = list(range(NC))
        self.hash = 0
        # touch[idx] = live lines through idx that are already started (>=1 slot
        # correct). touch[idx]==0 means idx is "cold": every placement there is
        # safe and carries no tactical content, so the search needs only one
        # representative cold cell instead of dozens of identical waiting moves.
        self.touch = [0] * NC
        # hot == {cell in empty : touch[cell] > 0}, maintained incrementally.
        # Scanning all 64 cells for this on every playout ply was 73% of the
        # engine's runtime; the set is typically 10-25 cells instead.
        self.hot = set()

    # -- setup ---------------------------------------------------------------
    def set_from_chars(self, board):
        """Load a grid of 'S' / 'O' / ' '. Order does not matter: the derived
        state (filled / dead / threats) is a pure function of the final board."""
        self.reset()
        for r in range(N):
            row = board[r]
            for c in range(N):
                ch = row[c]
                if ch == 'S':
                    self.make(r * N + c)
                elif ch == 'O':
                    self.make(NC + r * N + c)

    @property
    def nempty(self):
        return len(self.empty)

    # -- make / unmake -------------------------------------------------------
    # Both perform the same three steps, which is what makes them inverse:
    #   1. retract every threat carried by a line through idx,
    #   2. change the cell,
    #   3. re-assert every threat carried by a line through idx.
    # Step 1 is safe because a threat line has exactly one empty slot, so while
    # the board still holds its pre-step value that slot is unambiguous.
    def make(self, action):
        """Place the action's letter; return (points_scored, undo_token)."""
        idx = action & 63
        letter = S if action < NC else O
        bd, fl, dd = self.board, self.filled, self.dead
        tc, th, ln, tch = self.tcount, self.threats, self.lines, self.touch

        # Every threat line through idx points AT idx (a threat line has exactly
        # one empty slot, and idx is empty), so retracting needs no board lookup
        # and the whole update fits in one pass.
        bd[idx] = letter
        self.hot.discard(idx)
        hot = self.hot
        gained = 0
        for l, req in self.cl[idx]:
            f = fl[l]
            d = dd[l]
            if f == 2 and d == 0:
                sl = idx if req == S else NC + idx
                tc[sl] -= 1
                if tc[sl] == 0:
                    th.discard(sl)
            started = f > 0 and d == 0
            if req == letter:
                f += 1
                fl[l] = f
            else:
                d += 1
                dd[l] = d
            if started != (f > 0 and d == 0):
                a, b, c = ln[l]
                delta = -1 if started else 1
                for x in (a, b, c):
                    t = tch[x] + delta
                    tch[x] = t
                    if t == 0:
                        hot.discard(x)
                    elif t == 1 and delta == 1 and bd[x] == EMPTY:
                        hot.add(x)
            if f == 3:
                gained += 1
            elif f == 2 and d == 0:
                a, b, c = ln[l]
                sl = a if bd[a] == EMPTY else (NC + b if bd[b] == EMPTY else c)
                tc[sl] += 1
                if tc[sl] == 1:
                    th.add(sl)

        e, ep = self.empty, self.epos
        p = ep[idx]
        last = e.pop()
        if last != idx:
            e[p] = last
            ep[last] = p
        self.hash ^= ZOB[letter][idx]
        return gained, p

    def unmake(self, action, p):
        idx = action & 63
        letter = S if action < NC else O
        bd, fl, dd = self.board, self.filled, self.dead
        tc, th, ln, tch, hot = (self.tcount, self.threats, self.lines,
                               self.touch, self.hot)

        # Mirror image of make: retracting needs the board as it still stands
        # (idx filled, so a threat's empty slot is elsewhere), while every threat
        # re-asserted by the undo points at idx. So idx is cleared only at the
        # end and both directions stay single-pass.
        for l, req in self.cl[idx]:
            f = fl[l]
            d = dd[l]
            if f == 2 and d == 0:
                a, b, c = ln[l]
                sl = a if bd[a] == EMPTY else (NC + b if bd[b] == EMPTY else c)
                tc[sl] -= 1
                if tc[sl] == 0:
                    th.discard(sl)
            started = f > 0 and d == 0
            if req == letter:
                f -= 1
                fl[l] = f
            else:
                d -= 1
                dd[l] = d
            if started != (f > 0 and d == 0):
                a, b, c = ln[l]
                delta = -1 if started else 1
                for x in (a, b, c):
                    t = tch[x] + delta
                    tch[x] = t
                    if t == 0:
                        hot.discard(x)
                    elif t == 1 and delta == 1 and bd[x] == EMPTY:
                        hot.add(x)
            if f == 2 and d == 0:
                sl = idx if req == S else NC + idx
                tc[sl] += 1
                if tc[sl] == 1:
                    th.add(sl)
        bd[idx] = EMPTY
        if tch[idx]:
            hot.add(idx)

        e, ep = self.empty, self.epos
        if p == len(e):
            e.append(idx)
            ep[idx] = p
        else:
            moved = e[p]
            e.append(moved)
            ep[moved] = len(e) - 1
            e[p] = idx
            ep[idx] = p
        self.hash ^= ZOB[letter][idx]

    # -- queries -------------------------------------------------------------
    def points(self, action):
        return self.tcount[action]

    def threat_points(self):
        tc = self.tcount
        return sum(tc[a] for a in self.threats)

    def quiet_features(self, idx, letter):
        """(creates, kills, builds) for a non-scoring placement.

        creates -- live lines pushed to 2/3, i.e. threats handed to the opponent
        kills   -- started live lines this placement poisons for good
        builds  -- untouched live lines this placement opens at 1/3
        """
        creates = kills = builds = 0
        fl, dd = self.filled, self.dead
        for l, req in self.cl[idx]:
            if dd[l]:
                continue
            f = fl[l]
            if req == letter:
                if f == 1:
                    creates += 1
                elif f == 0:
                    builds += 1
            elif f:
                kills += 1
        return creates, kills, builds


def action_to_move(action):
    r, c = divmod(action & 63, N)
    return (r, c), ('S' if action < NC else 'O')


def move_to_action(rc, letter):
    r, c = rc
    return r * N + c + (0 if letter == 'S' else NC)


# ---------------------------------------------------------------------------
# Greedy playout, used as the search's leaf evaluator.
#
# A depth-limited search needs to answer "who wins the rest of the game?", and
# in SOS the rest of the game is a cascade fight worth tens of points -- so an
# "assume it is even" leaf value is worse than useless. A playout answers it
# concretely. The policy mirrors what a competent player does:
#   * always take the biggest available SOS (and keep taking: bonus turns),
#   * otherwise play the most *clustered* move that hands over no threat, which
#     preserves the far-from-the-action cells as waiting moves,
#   * spend a waiting move only when every clustered move is poisoned,
#   * if literally everything is poisoned, give away as little as possible.
# The playout mutates the position and then rewinds it exactly.
# ---------------------------------------------------------------------------

def pick_quiet(pos):
    """Best non-scoring placement under the playout policy.

    Hot spot: this runs once per quiet ply of every playout, so it does two
    things to stay cheap. It walks each cell's 12 lines a single time, deriving
    the stats for BOTH letters from that one pass and ranking with a packed
    integer key instead of tuples. And because the ranking's primary key is
    `touch`, it buckets the hot cells by touch and scans the buckets from the
    most clustered downwards: the best safe move must live in the highest bucket
    that contains one, so the scan stops there. That is an exact shortcut, not
    an approximation -- the move returned is identical to a full scan's.
    """
    touch, cl, fl, dd = pos.touch, pos.cl, pos.filled, pos.dead
    fallback = -1
    fb_key = 1 << 30
    hot = pos.hot
    if hot:
        ceiling = 1 << 30
        while True:
            # highest touch level still below `ceiling`, and the cells on it
            level = 0
            bucket = None
            for idx in hot:
                t = touch[idx]
                if t >= ceiling:
                    continue
                if t > level:
                    level, bucket = t, [idx]
                elif t == level:
                    bucket.append(idx)
            if not level:
                break
            ceiling = level
            best = -1
            best_key = -1
            for idx in bucket:
                # cs/bs/ks = creates/builds/kills if we place S here; co/bo/ko for O
                cs = bs = ks = co = bo = ko = 0
                for l, req in cl[idx]:
                    if dd[l]:
                        continue
                    f = fl[l]
                    if req == S:
                        if f == 1:
                            cs += 1
                        elif f == 0:
                            bs += 1
                        if f:
                            ko += 1          # an O here poisons this S-line
                    else:
                        if f == 1:
                            co += 1
                        elif f == 0:
                            bo += 1
                        if f:
                            ks += 1          # an S here poisons this O-line
                if cs:
                    k = (cs << 4) - ks
                    if k < fb_key:
                        fb_key, fallback = k, idx
                else:
                    k = (ks << 4) | bs
                    if k > best_key:
                        best_key, best = k, idx
                if co:
                    k = (co << 4) - ko
                    if k < fb_key:
                        fb_key, fallback = k, NC + idx
                else:
                    k = (ko << 4) | bo
                    if k > best_key:
                        best_key, best = k, NC + idx
            if best >= 0:
                return best
    for idx in pos.empty:                 # waiting move, far from the action
        if not touch[idx]:
            return idx
    return fallback                       # every move is poisoned


def rollout(pos):
    """Play to the end greedily; return the differential for the side to move
    at entry, leaving `pos` exactly as it was found."""
    tc = pos.tcount
    stack = []
    diff = 0
    sign = 1
    empty, threats = pos.empty, pos.threats
    make = pos.make
    while empty:
        if threats:
            mv = max(threats, key=tc.__getitem__)
            gained, tok = make(mv)
            diff += sign * gained
        else:
            mv = pick_quiet(pos)
            gained, tok = make(mv)
            sign = -sign                  # quiet move hands the turn over
        stack.append((mv, tok))
    unmake = pos.unmake
    for mv, tok in reversed(stack):
        unmake(mv, tok)
    return diff
