"""
distill_train.py  —  Expert Iteration / policy-value distillation for SOS.

WHY THIS INSTEAD OF tabula-rasa AlphaZero
-----------------------------------------
Your AlphaZero run showed the classic symptoms of the wrong tool for this game:
  * value loss collapsed to ~0 almost immediately (the score channels make
    "who is ahead" trivially predictable -> no real lookahead signal), and
  * self-play produced lopsided blow-outs (e.g. 53-0), so the net never learned
    nuanced defence.
SOS has a cheap, exact, local scoring rule, so a classical search (smart_bot.py)
already plays strongly. Expert Iteration exploits that: we let the STRONG search
bot teach a small network. The training targets are high quality from step one,
so this converges in a fraction of the time, on CPU, with no value collapse.

PIPELINE
--------
  1. Self-play games where SmartBot (the teacher) picks moves (with a little
     epsilon-randomness for state diversity). Record (state, teacher_move) and,
     at game end, the result as the value target.
  2. On-the-fly dihedral (8x) symmetry augmentation when batching — valid for
     both the standard and the toroidal board, for free data.
  3. Supervised training of a small AlphaZeroResNet:
        policy loss = cross-entropy(net, teacher_move)
        value  loss = MSE(net, game_result)

USAGE
-----
    python distill_train.py --games 150 --teacher-budget 0.4 \
        --blocks 4 --channels 64 --epochs 12

Outputs checkpoints to  checkpoints_distill/  (best.pth is the one neural_bot
loads). Run on Colab/GPU for the fastest training; data generation is CPU-bound
on the teacher regardless, so a lower --teacher-budget trades strength for speed.
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models import AlphaZeroResNet
from alpha_mcts import GameWrapper
from game_logic import SOSGame
from smart_bot import SmartBot

N = 8
ACTIONS = 128

# ── Dihedral symmetries of the board (cell remaps); valid on torus too ───────
def _sym_fns(n=N):
    return [
        lambda r, c: (r, c),               # identity
        lambda r, c: (c, n - 1 - r),       # rot90
        lambda r, c: (n - 1 - r, n - 1 - c),  # rot180
        lambda r, c: (n - 1 - c, r),       # rot270
        lambda r, c: (n - 1 - r, c),       # flip vertical
        lambda r, c: (r, n - 1 - c),       # flip horizontal
        lambda r, c: (c, r),               # transpose
        lambda r, c: (n - 1 - c, n - 1 - r),  # anti-transpose
    ]

SYMS = _sym_fns()


def _apply_sym_tensor(t, f, n=N):
    out = np.empty_like(t)
    for r in range(n):
        for c in range(n):
            rr, cc = f(r, c)
            out[:, rr, cc] = t[:, r, c]
    return out


def _apply_sym_action(a, f, n=N):
    sym = 0 if a < 64 else 64
    r, c = divmod(a % 64, n)
    rr, cc = f(r, c)
    return sym + rr * n + cc


def _char_board(int_board):
    return [[' ' if v == 0 else ('S' if v == 1 else 'O') for v in row]
            for row in int_board]


# ── 1. Data generation via the teacher ───────────────────────────────────────
def generate_data(num_games, teacher_budget, epsilon, wrap_around, verbose=True):
    teacher = SmartBot(wrap_around=wrap_around, time_budget=teacher_budget)
    examples = []  # (encoded_tensor np(6,8,8), action_int, player)
    t0 = time.perf_counter()

    for g in range(num_games):
        game = SOSGame(board_size=N, wrap_around=wrap_around)
        game.reset()
        per_game = []

        while not np.all(game.board != 0):
            state = {
                'board': np.copy(game.board),
                'scores': game.scores.copy(),
                'current_player': game.current_player,
                'sos_patterns': list(game.sos_patterns),
            }
            legal = game.get_valid_actions()

            if random.random() < epsilon:
                action = random.choice(legal)
            else:
                (r, c), letter = teacher.choose_move(_char_board(game.board))
                action = (r * N + c) + (0 if letter == 'S' else 64)
                if action not in legal:                # safety
                    action = random.choice(legal)

            enc = GameWrapper.encode_state(state).numpy().astype(np.float32)
            per_game.append((enc, action, game.current_player))
            game.step(action)

        s0, s1 = game.scores[0], game.scores[1]
        winner = 0 if s0 > s1 else (1 if s1 > s0 else -1)
        for enc, action, player in per_game:
            v = 0.0 if winner == -1 else (1.0 if player == winner else -1.0)
            examples.append((enc, action, v))

        if verbose and (g + 1) % 10 == 0:
            dt = time.perf_counter() - t0
            print(f"  game {g + 1}/{num_games}  "
                  f"({len(examples)} samples, {dt:.0f}s, scores {s0}-{s1})")

    return examples


# ── 2/3. Training with on-the-fly augmentation ───────────────────────────────
def make_batch(examples, batch_size, device):
    xs = np.empty((batch_size, 6, N, N), dtype=np.float32)
    pis = np.zeros((batch_size, ACTIONS), dtype=np.float32)
    vs = np.empty((batch_size,), dtype=np.float32)
    for i in range(batch_size):
        enc, action, v = random.choice(examples)
        f = random.choice(SYMS)
        xs[i] = _apply_sym_tensor(enc, f)
        pis[i, _apply_sym_action(action, f)] = 1.0
        vs[i] = v
    return (torch.from_numpy(xs).to(device),
            torch.from_numpy(pis).to(device),
            torch.from_numpy(vs).to(device))


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    print(f"Device: {device}")

    print(f"Generating {args.games} teacher games "
          f"(budget {args.teacher_budget}s/move)...")
    examples = generate_data(args.games, args.teacher_budget, args.epsilon,
                             not args.no_wrap)
    print(f"Collected {len(examples)} base samples (x8 via symmetry).")

    net = AlphaZeroResNet(N, args.blocks, args.channels, input_channels=6).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    steps_per_epoch = max(1, len(examples) // args.batch)
    log_path = os.path.join(args.out, 'distill_log.csv')
    with open(log_path, 'w', newline='') as fh:
        csv.writer(fh).writerow(['epoch', 'policy_loss', 'value_loss', 'total'])

    net.train()
    for epoch in range(args.epochs):
        pl, vl = [], []
        for _ in range(steps_per_epoch):
            x, pi, v = make_batch(examples, args.batch, device)
            out_pi, out_v = net(x)
            l_pi = -torch.mean(torch.sum(pi * F.log_softmax(out_pi, dim=1), dim=1))
            l_v = F.mse_loss(out_v.view(-1), v)
            loss = l_pi + l_v
            opt.zero_grad()
            loss.backward()
            opt.step()
            pl.append(l_pi.item())
            vl.append(l_v.item())

        a_pl, a_vl = float(np.mean(pl)), float(np.mean(vl))
        print(f"Epoch {epoch + 1}/{args.epochs}  "
              f"PolicyLoss {a_pl:.4f}  ValueLoss {a_vl:.4f}")
        with open(log_path, 'a', newline='') as fh:
            csv.writer(fh).writerow([epoch + 1, a_pl, a_vl, a_pl + a_vl])

        torch.save({'iteration': epoch + 1,
                    'model_state_dict': net.state_dict(),
                    'config': {'blocks': args.blocks, 'channels': args.channels}},
                   os.path.join(args.out, f'model_{epoch + 1}.pth'))

    torch.save({'iteration': args.epochs,
                'model_state_dict': net.state_dict(),
                'config': {'blocks': args.blocks, 'channels': args.channels}},
               os.path.join(args.out, 'best.pth'))
    print(f"Done. best.pth written to {args.out}/  "
          f"(neural_bot.py will pick it up automatically).")


def main():
    p = argparse.ArgumentParser(description='Expert-Iteration distillation for SOS')
    p.add_argument('--games', type=int, default=150)
    p.add_argument('--teacher-budget', type=float, default=0.4,
                   help='seconds/move for the SmartBot teacher (strength vs speed)')
    p.add_argument('--epsilon', type=float, default=0.12,
                   help='fraction of random moves for state diversity')
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', type=str, default='checkpoints_distill')
    p.add_argument('--no-wrap', action='store_true', help='train for standard (non-toroidal) board')
    train(p.parse_args())


if __name__ == "__main__":
    main()
