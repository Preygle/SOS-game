"""
neural_bot.py  —  Inference wrapper for the Expert-Iteration network.

Drop-in compatible with smart_bot / greedy_bot:

    bot = NeuralBot(wrap_around=True)
    if bot.available:
        (r, c), letter = bot.choose_move(board)   # board: ' ' / 'S' / 'O'

Design
------
This loads a *small* policy/value net trained by distill_train.py (Expert
Iteration: the net imitates the strong SmartBot teacher). At play time it:

  1. runs one forward pass to get a move-probability distribution,
  2. ALWAYS takes a free SOS if one exists (tactical safety net), otherwise
  3. plays the network's highest-probability legal move.

If torch isn't importable (your game's Python had a torch DLL issue) or no
checkpoint is found, `available` stays False and the caller falls back to the
classical SmartBot — the game never crashes over a missing model.

Encoding is delegated to alpha_mcts.GameWrapper.encode_state so it is byte-for-
byte identical to what the network was trained on.
"""

import os

# Checkpoints to try, best/newest first. Architecture is read from the saved
# 'config' key when present (distill_train.py writes it); otherwise we fall back
# to trying these (blocks, channels) shapes.
_PATHS = ["checkpoints_distill/best.pth", "checkpoints_v3/best.pth",
          "checkpoints/best.pth"]
_FALLBACK_SHAPES = [(4, 64), (6, 128)]

S_CH, O_CH, EMPTY_CH = 'S', 'O', ' '


class NeuralBot:
    def __init__(self, wrap_around=True, tactical=True):
        self.wrap_around = wrap_around
        self.tactical = tactical
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

        for path in _PATHS:
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

    # ── public API ──────────────────────────────────────────────────────────
    def choose_move(self, board):
        torch = self._torch
        legal = self._legal_actions(board)
        if not legal:
            return (0, 0), 'S'

        # 1) Tactical safety net: never pass up a free SOS.
        if self.tactical:
            best_a, best_s = None, 0
            for a in legal:
                s = self._immediate_score(board, a)
                if s > best_s:
                    best_s, best_a = s, a
            if best_a is not None:
                return self._to_move(best_a)

        # 2) Otherwise follow the network policy.
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

        best_a, best_p = legal[0], -1.0
        for a in legal:
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
