"""
neural_bot.py  —  Inference wrapper for the Expert-Iteration network.

Drop-in compatible with smart_bot / greedy_bot:

    bot = NeuralBot(wrap_around=True)
    if bot.available:
        (r, c), letter = bot.choose_move(board)   # board: ' ' / 'S' / 'O'

Design
------
STATUS: not wired into the game. Measured over 6 games each, this bot went
3-3 against greedy_bot (14.8 points to 23.8) and 2-4 against the old SmartBot,
which made it the weakest real bot in the project -- and it used to sit behind
the menu entry that promises the strongest opponent. The game now runs
strong_bot.StrongBot in both slots. This file still works and still auto-loads
the newest checkpoint, so distillation experiments remain runnable; it is just
no longer what you play against. See docs/AI.md for the measurements.

This loads a *small* policy/value net trained by distill_train.py (Expert
Iteration: the net imitates the strong search teacher). At play time it:

  1. runs one forward pass to get a move-probability distribution,
  2. ALWAYS takes a free SOS if one exists (tactical safety net), otherwise
  3. plays the network's highest-probability legal move.

If torch isn't importable (your game's Python had a torch DLL issue) or no
checkpoint is found, `available` stays False and the caller falls back to the
classical SmartBot — the game never crashes over a missing model.

Encoding is delegated to alpha_mcts.GameWrapper.encode_state so it is byte-for-
byte identical to what the network was trained on.
"""

import glob
import os

# Stable checkpoint locations, tried after the newest timestamped run.
_STABLE_PATHS = ["checkpoints_distill/best.pth", "checkpoints_v3/best.pth",
                 "checkpoints/best.pth"]
_FALLBACK_SHAPES = [(4, 64), (6, 128)]


def _resolve_checkpoints(base="."):
    """Newest ``checkpoints_distill_<timestamp>/best.pth`` first, then stable paths.

    run_full_training.bat / run_distill.bat write each run to its own
    timestamped folder (so old data stays intact). The folder name is
    YYYYMMDD_HHMMSS, which sorts chronologically, so a reverse sort puts the
    most recent trained model first. Without this, a fresh training run would
    never be picked up and the game silently fell back to the classical bot.
    """
    ts = sorted(glob.glob(os.path.join(base, "checkpoints_distill_*", "best.pth")),
                reverse=True)
    seen, out = set(), []
    for p in ts + _STABLE_PATHS:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

S_CH, O_CH, EMPTY_CH = 'S', 'O', ' '


class NeuralBot:
    def __init__(self, wrap_around=True, tactical=True, defensive=True):
        self.wrap_around = wrap_around
        self.tactical = tactical
        self.defensive = defensive
        self.available = False
        self._torch = None
        self._model = None
        self._encode = None
        self._counter = None
        self._try_load()

    def _try_load(self):
        try:
            import torch
            from models import AlphaZeroResNet
            from alpha_mcts import GameWrapper
            from smart_bot import SmartBot
        except Exception as e:
            print(f"[NeuralBot] torch / deps unavailable: {e}")
            return

        for path in _resolve_checkpoints():
            if not os.path.exists(path):
                continue
            try:
                ckpt = torch.load(path, map_location="cpu")
            except Exception as e:
                print(f"[NeuralBot] Could not read {path}: {e}")
                continue

            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and \
                "model_state_dict" in ckpt else ckpt
            cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
            shapes = [(cfg["blocks"], cfg["channels"])] if cfg else _FALLBACK_SHAPES

            for blocks, ch in shapes:
                try:
                    model = AlphaZeroResNet(8, blocks, ch, input_channels=6)
                    model.load_state_dict(state)
                    model.eval()
                    self._torch = torch
                    self._model = model
                    self._encode = GameWrapper.encode_state
                    self._counter = SmartBot(wrap_around=self.wrap_around,
                                             time_budget=0.0)
                    self.available = True
                    print(f"[NeuralBot] Loaded {path} ({blocks}x{ch}).")
                    return
                except Exception as e:
                    print(f"[NeuralBot] {path} not a {blocks}x{ch} net: {e}")
        print("[NeuralBot] No compatible checkpoint found.")

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _to_int_board(board):
        import numpy as np
        n = len(board)
        b = np.zeros((n, n), dtype=int)
        for r in range(n):
            for c in range(n):
                ch = board[r][c]
                if ch == S_CH:
                    b[r, c] = 1
                elif ch == O_CH:
                    b[r, c] = 2
        return b

    def _legal_actions(self, board):
        n = len(board)
        acts = []
        for r in range(n):
            for c in range(n):
                if board[r][c] == EMPTY_CH:
                    idx = r * n + c
                    acts.append(idx)        # S
                    acts.append(64 + idx)   # O
        return acts

    def _immediate_score(self, board, action):
        """SOS formed by `action`, using SmartBot's exact toroidal counter."""
        n = len(board)
        idx = action % 64
        piece = 1 if action < 64 else 2
        self._counter.wrap_around = self.wrap_around
        if self._counter._wrap_cached != self.wrap_around:
            self._counter._build_neighbours()
        # Build a flat int board for the counter.
        flat = [0] * (n * n)
        for r in range(n):
            for c in range(n):
                ch = board[r][c]
                flat[r * n + c] = 1 if ch == S_CH else 2 if ch == O_CH else 0
        return self._counter._count_sos(flat, idx, piece)

    def _opp_best_after(self, board, action):
        """Best SOS the opponent could score on their reply if we play `action`.

        Only meaningful when `action` itself scores nothing (otherwise we'd keep
        the turn), which is exactly the situation the policy branch handles.
        """
        r, c = divmod(action % 64, 8)
        piece = S_CH if action < 64 else O_CH
        nb = [row[:] for row in board]
        nb[r][c] = piece
        best = 0
        for b in self._legal_actions(nb):
            s = self._immediate_score(nb, b)
            if s > best:
                best = s
        return best

    # ── public API ──────────────────────────────────────────────────────────
    def choose_move(self, board):
        torch = self._torch
        legal = self._legal_actions(board)
        if not legal:
            return (0, 0), 'S'

        # 1) Tactical safety net: never pass up a free SOS (take the biggest).
        if self.tactical:
            best_a, best_s = None, 0
            for a in legal:
                s = self._immediate_score(board, a)
                if s > best_s:
                    best_s, best_a = s, a
            if best_a is not None:
                return self._to_move(best_a)

        # 2) Defensive filter: no move scores here, so whatever we play, the
        #    opponent replies. Keep only the moves that give them the smallest
        #    free SOS. This is what stops the net from gifting the opponent
        #    setups (it was losing ~5-52 to the greedy bot without it).
        candidates = legal
        filled = sum(1 for row in board for ch in row if ch != EMPTY_CH)
        if self.defensive and len(legal) > 1 and filled >= 2:
            risk = {a: self._opp_best_after(board, a) for a in legal}
            m = min(risk.values())
            candidates = [a for a in legal if risk[a] == m]

        # 3) Among the safe candidates, follow the network policy.
        int_board = self._to_int_board(board)
        state = {
            'board': int_board,
            'scores': {0: 0, 1: 0},
            'current_player': 1,      # the bot plays as P2
            'sos_patterns': [],
        }
        with torch.no_grad():
            x = self._encode(state).unsqueeze(0)        # (1, 6, 8, 8)
            logits, _value = self._model(x)
            probs = torch.softmax(logits, dim=1)[0]

        best_a, best_p = candidates[0], -1.0
        for a in candidates:
            p = probs[a].item()
            if p > best_p:
                best_p, best_a = p, a
        return self._to_move(best_a)

    @staticmethod
    def _to_move(action):
        idx = action % 64
        r, c = divmod(idx, 8)
        return (r, c), ('S' if action < 64 else 'O')


if __name__ == "__main__":
    bot = NeuralBot(wrap_around=True)
    print("available:", bot.available)
    if bot.available:
        blank = [[' '] * 8 for _ in range(8)]
        blank[3][3] = 'S'; blank[3][5] = 'S'
        print(bot.choose_move(blank))
