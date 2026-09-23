"""
eval_bots.py -- measure how strong the trained NeuralBot actually plays.

Runs head-to-head matches (alternating who moves first) of the neural net vs
Random, the greedy bot, and the classical SmartBot deep search, and prints
win/loss/draw plus average score. Use this to decide whether a freshly trained
net is worth enabling -- if it does not beat SmartBot, the game's SmartBot
fallback is still the stronger opponent.

    python eval_bots.py                 # quick: 20 games each, SmartBot budget 0.3s
    python eval_bots.py --games 50 --smart-budget 1.0
    python eval_bots.py --checkpoint checkpoints_distill_20260723_192749/best.pth
"""

import argparse
import random

import numpy as np

from game_logic import SOSGame
from greedy_bot import SOSBot as GreedyBot
from smart_bot import SmartBot
from strong_bot import StrongBot
from neural_bot import NeuralBot


def _char_board(bi):
    return [[' ' if v == 0 else ('S' if v == 1 else 'O') for v in row] for row in bi]


class RandomBot:
    name = "random"

    def choose_move(self, board):
        legal = [(r, c) for r in range(8) for c in range(8) if board[r][c] == ' ']
        r, c = random.choice(legal)
        return (r, c), random.choice(['S', 'O'])


def _action(move):
    (r, c), L = move
    return (r * 8 + c) + (0 if L == 'S' else 64)


def play_match(bot_a, bot_b, n_games, wrap=True, seed=0):
    random.seed(seed)
    wins = [0, 0, 0]           # a-wins, b-wins, draws
    tot_a = tot_b = 0
    for i in range(n_games):
        game = SOSGame(8, wrap)
        game.reset()
        seats = {i % 2: bot_a, 1 - i % 2: bot_b}
        while not np.all(game.board != 0):
            mover = seats[game.current_player]
            game.step(_action(mover.choose_move(_char_board(game.board))))
        me, op = game.scores[i % 2], game.scores[1 - i % 2]
        tot_a += me
        tot_b += op
        wins[0 if me > op else 1 if op > me else 2] += 1
    return wins, tot_a / n_games, tot_b / n_games


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=20)
    p.add_argument('--smart-budget', type=float, default=0.3)
    p.add_argument('--checkpoint', type=str, default=None,
                   help='force a specific best.pth (else newest is auto-found)')
    p.add_argument('--no-wrap', action='store_true')
    args = p.parse_args()
    wrap = not args.no_wrap

    if args.checkpoint:
        import neural_bot
        neural_bot._resolve_checkpoints = lambda base=".": [args.checkpoint]

    net = NeuralBot(wrap_around=wrap)
    if not net.available:
        print("NeuralBot is NOT available -- no checkpoint loaded. Train first.")
        return

    opponents = [
        RandomBot(),
        GreedyBot(wrap_around=wrap),
        SmartBot(wrap_around=wrap, time_budget=args.smart_budget),
        StrongBot(wrap_around=wrap, time_budget=args.smart_budget),
    ]
    labels = ["random", "greedy", f"SmartBot({args.smart_budget}s)",
              f"StrongBot({args.smart_budget}s)"]

    print(f"\nNeuralBot over {args.games} games each (alternating first move):\n")
    print(f"  {'opponent':<16} {'W-L-D':>10} {'net avg':>9} {'opp avg':>9}  result")
    print("  " + "-" * 58)
    for opp, label in zip(opponents, labels):
        (w, l, d), a, b = play_match(net, opp, args.games, wrap)
        verdict = "WIN " if w > l else "loss" if l > w else "draw"
        print(f"  {label:<16} {f'{w}-{l}-{d}':>10} {a:>9.1f} {b:>9.1f}  {verdict}")
    print()


if __name__ == "__main__":
    main()
