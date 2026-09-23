"""
laya_bot.py  --  an SOS opponent whose move choice is made by Laya.

    bot = LayaBot(wrap_around=True)
    (r, c), letter = bot.choose_move(board)      # board: list[list] of ' '/'S'/'O'

WHAT LAYA CAN AND CANNOT DO HERE
--------------------------------
Laya is a System One decision model: it never generates text, it answers *typed
questions* about a state and returns calibrated probabilities. A choice question
picks one label from a set, and its accuracy falls off hard past ~20 labels (the
model card reports 0.425 on a 77-label task). SOS offers up to 128 moves -- 64
cells times two letters -- so handing Laya the raw move list would be asking it
to do the thing it is worst at.

So the work is split the way these game agents are normally built:

  * the game engine (sos_engine.py) enumerates legal moves, discards the
    thousands of equivalent waiting moves, and annotates a handful of genuinely
    different candidates with what each one does;
  * Laya makes the actual decision, picking one candidate from <= 8 labels.

The engine does no lookahead. It reports only what is true after one placement:
how many points the move scores, how big a chain it starts, and what the
opponent can score in reply. Choosing between those trade-offs -- take two
points now, or take none and leave the opponent nothing -- is Laya's job, and
that is the part of SOS that actually decides games.

HONESTY ABOUT WHAT THIS PROVES
------------------------------
Because the shortlist is annotated, a bot that just read the numbers would also
play reasonably. `policy="first"` and `policy="random"` pick from the identical
shortlist without calling the model, so you can measure what Laya adds rather
than assume it. See arena.py and docs/LAYA_BOT.md for measured results.
"""

import json
import os
import subprocess
import sys

from sos_engine import NC, S, O, Position, action_to_move
from strong_bot import StrongBot

# The Laya environment (transformers 5.x) is deliberately separate from this
# project's (transformers 4.50), so the model runs in its own interpreter.
LAYA_PYTHON = os.environ.get("LAYA_PYTHON", r"D:\laya\env\Scripts\python.exe")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laya_worker.py")

FILES = "ABCDEFGH"

# Kept short on purpose: instructions and option labels share one token budget
# (head_max_len), so a long preamble here would truncate the move descriptions.
INSTRUCTIONS = (
    "You are playing SOS and it is your move. Each option says what it does. "
    "Pick the move that wins the most points over the whole game, not just now. "
    "Giving the opponent a big chain loses games; a small safe move often wins them."
)

# The deep version of the rules goes in the STATE, which has a much larger budget
# (1024 tokens on the multilingual checkpoint) than the question head.
RULES = """SOS RULES.
Board: 8x8 grid, 64 cells, starts empty. Two players.
A turn: place one letter, S or O, into any empty cell. You choose both the cell and the letter.
Scoring: if your placement completes the pattern S-O-S in three consecutive cells in a line, you score 1 point for each S-O-S you complete with that one placement (a single placement can complete two or three at once). Lines run horizontally, vertically, and on both diagonals.
BONUS TURN: if your placement scores, you keep the turn and move again immediately. You can keep scoring move after move. If your placement scores nothing, the turn passes to the opponent.
End: the game ends when all 64 cells are full. Highest score wins; equal scores draw.
{wrap}
WHAT MAKES A MOVE GOOD.
A "threat" is an empty cell where somebody can score right now. Two S's with one empty cell between them is a threat: an O in that gap scores. An S next to an O with the far cell empty is a threat: an S in that cell scores.
Because scoring grants another turn, threats come in chains: whoever moves into a dense board first can score repeatedly and sweep many points in one unbroken run.
So the game is usually decided by who runs out of safe moves. A "safe" move creates no threat for the opponent. While safe moves exist both players play them. When a player has none left, they are forced to open the board and hand the opponent a chain.
Therefore: take points when they are free, but do not create threats you cannot use, and keep safe waiting moves in reserve for the endgame."""

WRAP_NOTE = ("Orbit mode is ON: the board is a torus. The left edge joins the right edge, "
             "the top joins the bottom, and diagonals wrap around corners. Lines continue "
             "across every edge.")
NOWRAP_NOTE = "Orbit mode is OFF: lines stop at the edges of the board."


def square(idx):
    """Board index -> the game's own chess-style name, e.g. 27 -> 'D5'."""
    r, c = divmod(idx, 8)
    return "%s%d" % (FILES[c], 8 - r)


class LayaBot:
    def __init__(self, wrap_around=True, model="typed-decisions", policy="laya",
                 shortlist=8, mode="search", rules=False, verbose=False,
                 time_budget=4.0, margin=0):
        self.wrap_around = wrap_around
        self.model = model
        # "search"   -> alpha-beta ranks every root move; Laya chooses among the
        #               moves the search cannot separate. Default, and the only
        #               mode that plays properly: the search takes free points
        #               and sees the endgame, so Laya can never throw either away.
        # "strategy" -> one choice question over a few named plans (pure Laya)
        # "move"     -> one choice question over <= shortlist concrete moves
        # The last two are the unaided experiment; they play very badly. See
        # laya_probe.py and docs/LAYA_BOT.md for the numbers.
        self.mode = mode
        self.time_budget = time_budget
        # How many points Laya is allowed to give up against the search's own
        # best value. 0 means it may only pick among exact ties, so the bot is
        # never weaker than the search itself.
        self.margin = margin
        self.search = StrongBot(wrap_around=wrap_around,
                                time_budget=time_budget) if mode == "search" else None
        self.last_depth = 0
        self.last_tied = 0
        # Putting the full rules in the state measured no better than leaving
        # them out, and cost 2-3x the latency, so it is off by default.
        self.rules = rules
        self.policy = policy            # "laya" | "first" | "random"
        self.shortlist = shortlist
        self.verbose = verbose
        self.pos = Position(wrap_around)
        self._proc = None
        self.available = policy != "laya"
        self.last_ms = 0.0
        self.last_confidence = None
        self.last_candidates = []
        self.last_choice = None
        self.fallbacks = 0
        self.calls = 0
        if policy == "laya":
            self._start()

    # -- worker process ------------------------------------------------------
    def _start(self):
        if not os.path.exists(LAYA_PYTHON):
            print("[LayaBot] no Laya interpreter at %s; set LAYA_PYTHON." % LAYA_PYTHON)
            return
        try:
            self._proc = subprocess.Popen(
                [LAYA_PYTHON, "-u", WORKER, "--model", self.model],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", bufsize=1)
            hello = json.loads(self._proc.stdout.readline())
            if hello.get("ready"):
                self.available = True
                print("[LayaBot] %s checkpoint ready (%ss)." % (hello["model"], hello["load_s"]))
        except Exception as e:
            print("[LayaBot] could not start worker: %s" % e)

    def _ask(self, state, questions):
        if self._proc is None or self._proc.poll() is not None:
            return None
        try:
            self._proc.stdin.write(json.dumps({"state": state, "questions": questions}) + "\n")
            self._proc.stdin.flush()
            reply = json.loads(self._proc.stdout.readline())
        except Exception:
            return None
        if not reply.get("ok"):
            return None
        self.last_ms = reply.get("ms", 0.0)
        return reply["answers"]

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write('{"cmd": "quit"}\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    # -- engine side: legal, annotated, de-duplicated candidates --------------
    def _chain_and_reply(self, action):
        """What the move actually does: (points now, points if the chain is run
        out greedily, best single SOS the opponent can score in reply)."""
        pos = self.pos
        now, tok = pos.make(action)
        undo = [(action, tok)]
        chain = now
        if now:                       # a scoring move keeps the turn: follow it
            while pos.threats:
                nxt = max(pos.threats, key=pos.tcount.__getitem__)
                got, t2 = pos.make(nxt)
                undo.append((nxt, t2))
                chain += got
        reply = max((pos.tcount[a] for a in pos.threats), default=0)
        for a, t in reversed(undo):
            pos.unmake(a, t)
        return now, chain, reply

    def _candidates(self):
        """<= self.shortlist genuinely different moves, each with a short label.

        No search: every number in a description is a one-placement fact."""
        pos = self.pos
        tc, touch = pos.tcount, pos.touch
        scoring = sorted(pos.threats, key=lambda a: -tc[a])[:3]

        safe, unsafe, cold = [], [], -1
        for idx in pos.empty:
            if not touch[idx]:
                if cold < 0:
                    cold = idx            # all cold cells are the same move
                continue
            for base, letter in ((0, S), (NC, O)):
                a = base + idx
                if tc[a]:
                    continue              # already listed as a scoring move
                creates, kills, builds = pos.quiet_features(idx, letter)
                (unsafe if creates else safe).append((creates, -kills, -builds, a))
        safe.sort()
        unsafe.sort()

        picked = list(scoring)
        picked += [t[-1] for t in safe[:3]]
        if cold >= 0:
            picked.append(cold)           # the waiting move, far from the action
        picked += [t[-1] for t in unsafe[:2]]
        if not picked:                    # no empty cell is interesting: take any
            picked = [pos.empty[0]]
        picked = list(dict.fromkeys(picked))[:self.shortlist]

        criteria, actions = {}, {}
        for a in picked:
            now, chain, reply = self._chain_and_reply(a)
            label = "%s-%s" % (square(a & 63), "S" if a < NC else "O")
            if now:
                desc = "scores %d now, %d with the chain, then opponent scores %d" % (
                    now, chain, reply)
            elif a == cold:
                desc = "scores 0, waiting move far from play, opponent scores %d" % reply
            else:
                desc = "scores 0, opponent then scores %d" % reply
            criteria[label] = desc
            actions[label] = a
        return criteria, actions

    def _plans(self):
        """Named plans, each bound to the concrete move the engine would play.

        A choice question over four semantic labels is the shape Laya was built
        for; an eight-way comparison of numeric move descriptions is not."""
        pos = self.pos
        tc, touch = pos.tcount, pos.touch
        plans, actions = {}, {}

        if pos.threats:
            best = max(pos.threats, key=tc.__getitem__)
            now, chain, reply = self._chain_and_reply(best)
            plans["take_points"] = ("score %d point(s) now at %s and keep the turn, "
                                    "%d with the chain" % (now, square(best & 63), chain))
            actions["take_points"] = best

        safe, unsafe, cold = [], [], -1
        for idx in pos.empty:
            if not touch[idx]:
                if cold < 0:
                    cold = idx
                continue
            for base, letter in ((0, S), (NC, O)):
                a = base + idx
                if tc[a]:
                    continue
                creates, kills, builds = pos.quiet_features(idx, letter)
                (unsafe if creates else safe).append((creates, -kills, -builds, a))
        safe.sort()
        unsafe.sort()

        if safe:
            a = safe[0][-1]
            plans["play_safe"] = ("place at %s, scoring nothing but giving the "
                                  "opponent no way to score" % square(a & 63))
            actions["play_safe"] = a
        if cold >= 0:
            plans["wait"] = ("place at %s, far from the action, keeping the "
                             "position closed" % square(cold))
            actions["wait"] = cold
        if unsafe and len(plans) < 4:
            a = unsafe[0][-1]
            _n, _c, reply = self._chain_and_reply(a)
            plans["open_up"] = ("place at %s, which lets the opponent score %d"
                                % (square(a & 63), reply))
            actions["open_up"] = a
        if not plans:
            a = pos.empty[0]
            plans["any"] = "place at %s" % square(a)
            actions["any"] = a
        return plans, actions

    def _searched(self):
        """Moves the search rates within `margin` of its best, annotated.

        These are the moves the engine considers equivalent, so any of them is a
        fine choice -- which is exactly the decision worth handing to Laya."""
        ranked = self.search.last_results
        self.last_depth = self.search.last_depth
        if not ranked:
            # The search short-circuits some positions (an empty board) without
            # ranking anything. Fall back to the unaided shortlist rather than
            # taking the whole game down.
            return self._plans()
        best = ranked[0][0]
        tied = [a for v, a in ranked if v >= best - self.margin]
        self.last_tied = len(tied)
        criteria, actions = {}, {}
        for a in tied[:self.shortlist]:
            now, chain, reply = self._chain_and_reply(a)
            label = "%s-%s" % (square(a & 63), "S" if a < NC else "O")
            if now:
                desc = "scores %d now, %d with the chain, then opponent scores %d" % (
                    now, chain, reply)
            else:
                desc = "scores 0, opponent then scores %d" % reply
            criteria[label] = desc
            actions[label] = a
        return criteria, actions

    # -- state text ----------------------------------------------------------
    def _render(self, board, my_score, opp_score):
        """Board plus score. The full rules are prepended only if self.rules."""
        rows = []
        for r in range(8):
            rows.append("%d %s" % (8 - r, " ".join(
                ch if ch != ' ' else '.' for ch in board[r])))
        rows.append("  " + " ".join(FILES))
        grid = "\n".join(rows)
        empties = sum(row.count(' ') for row in board)
        rules = RULES.format(wrap=WRAP_NOTE if self.wrap_around else NOWRAP_NOTE)
        return ("%s\n\nCURRENT GAME.\nBoard (a dot is an empty cell):\n%s\n\n"
                "Your score %d, opponent score %d. %d empty cells left."
                % (rules, grid, my_score, opp_score, empties))

    # -- public API ----------------------------------------------------------
    def choose_action(self, board, my_score=0, opp_score=0):
        pos = self.pos
        if pos.wrap != self.wrap_around:
            pos.set_topology(self.wrap_around)
        pos.set_from_chars(board)
        if not pos.empty:
            return 0

        if self.mode == "search":
            # Run the search first: it fixes both things a bare Laya bot gets
            # wrong -- it never misses a free score, and it sees the endgame out.
            self.search.wrap_around = self.wrap_around
            self.search.time_budget = self.time_budget
            self.search.choose_action(board)
            criteria, actions = self._searched()
        elif self.mode == "strategy":
            criteria, actions = self._plans()
        else:
            criteria, actions = self._candidates()
        labels = list(criteria)
        self.last_candidates = [(l, criteria[l]) for l in labels]
        self.last_confidence = None

        if len(labels) == 1:
            self.last_choice = labels[0]
            return actions[labels[0]]
        if self.policy == "first":
            self.last_choice = labels[0]
            return actions[labels[0]]
        if self.policy == "random":
            import random
            self.last_choice = random.choice(labels)
            return actions[self.last_choice]

        self.calls += 1
        answers = self._ask(
            self._render(board, my_score, opp_score),
            {"move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}})

        pick = None
        if answers is not None:
            ans = answers.get("move", {})
            pick = ans.get("choice")
            self.last_confidence = ans.get("confidence")
        if pick not in actions:               # worker died, or a label came back odd
            self.fallbacks += 1
            pick = labels[0]
        self.last_choice = pick
        if self.verbose:
            print("[LayaBot] %s (conf %.2f, %.0f ms) from %s"
                  % (pick, self.last_confidence or 0.0, self.last_ms, labels))
        return actions[pick]

    def ask_choice(self, state, criteria):
        """Put one choice question to Laya and return the winning label.

        Used by mc_bot.py, which builds its own options from simulation, so this
        deliberately adds no candidate generation of its own."""
        answers = self._ask(state, {"move": {"type": "choice",
                                             "instructions": INSTRUCTIONS,
                                             "criteria": criteria}})
        if answers is None:
            return None
        ans = answers.get("move", {})
        self.last_confidence = ans.get("confidence")
        return ans.get("choice")

    def choose_move(self, board, my_score=0, opp_score=0):
        return action_to_move(self.choose_action(board, my_score, opp_score))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    policy = sys.argv[1] if len(sys.argv) > 1 else "laya"
    bot = LayaBot(wrap_around=True, policy=policy, verbose=True)
    b = [[' '] * 8 for _ in range(8)]
    b[3][3] = 'S'; b[3][5] = 'S'; b[4][4] = 'O'; b[2][2] = 'S'
    print("move:", bot.choose_move(b))
    print("candidates:")
    for label, desc in bot.last_candidates:
        print("   %-6s %s" % (label, desc))
    bot.close()
