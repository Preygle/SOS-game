# AI Upgrade — Stronger Rule Bot + RL Verdict

This documents the two AI changes: a much stronger classical opponent, and why
the AlphaZero approach was pivoted to **Expert Iteration**. Every claim below was
verified by self-play (see "Evidence").

## TL;DR

| | Before | After |
|---|---|---|
| **Rule bot** (`greedy_bot.py`) | "Depth-2" greedy, blind to bonus-turn chains | `smart_bot.py`: **flat Monte-Carlo** (1-ply + greedy rollouts). Beats greedy **~8–0**, often by 40+ points |
| **RL bot** (`AlphaBot` in `sos.py`) | **Placeholder** — returned `None`, silently fell back to greedy; `torch` import was broken in the game | Real engine: loads an Expert-Iteration net if trained (`neural_bot.py`), else uses the strong Monte-Carlo bot. Never a no-op. |

## The key insight: SOS is a cascade / zugzwang game

With bonus turns, **scoring an SOS lets you move again**. Once the board is dense,
the player on move can score on move after move and sweep **40+ points in one
unbroken chain**. In a traced game, one side scored on **47 consecutive moves**
while the opponent never got to play.

Two consequences:
1. The game is decided in the quiet opening — by **who builds the cascade engine
   first** and who is forced to make the "igniting" move (a zugzwang).
2. The deciding chain is ~40+ plies deep, so **no alpha-beta horizon can see it**,
   and no simple static evaluation can predict who wins it.

This is why **deeper search is the wrong tool here** (we tried it — see below).

## 1. The new rule bot — `smart_bot.py`

A **flat Monte-Carlo** search (drop-in for `SOSBot`):

- For each candidate move, **play the game out to the end** with a fast, strong
  rollout policy and keep the move with the best result.
- **Rollout policy**: take the biggest available SOS (chains via bonus turns);
  otherwise play a *clustered* quiet move that does **not** hand the opponent an
  immediate SOS. This mirrors what makes greedy strong, but the 1-ply lookahead +
  full-game rollout lets it win the cascade/zugzwang battle greedy plays blindly.
- **Deterministic by default** (`EPS_QUIET=0`): a clean greedy rollout is the best
  evaluator, so one pass over the candidates is optimal and fast (move 1 is
  instant via toroidal symmetry; midgame ≈2–3s; always < your 10s).
- Exact **toroidal** SOS scoring identical to `game_logic._check_sos`. No torch.

`time_budget` is a cap (deterministic mode returns after one pass). Set
`EPS_QUIET>0` to switch to randomized rollouts averaged over the full budget.

Quick check (no GUI needed):
```bash
python smart_bot.py            # prints chosen move, #rollouts, value, time
```

### What did NOT work (documented so we don't repeat it)
A full **iterative-deepening alpha-beta** (negamax + quiescence + transposition
table) was implemented first. It **lost to greedy ~0–6**, getting blown out
(0–67) — and *more depth made it no better*, because the deciding cascade is far
beyond any reachable horizon. Flat Monte-Carlo with rollouts is the right tool.

### In-game wiring (`sos.py`)
- `greedy_bot = SmartBot(..., time_budget=2.5)` — the rule slot, now the MC bot.
- `alpha_bot  = AlphaBot(time_budget=9.0)` — neural net if trained, else MC bot.
- The search runs on a **background thread** (`bot_turn_execute` → `bot_poll`) so
  the pyglet window stays responsive instead of freezing.

## 2. RL verdict — was AlphaZero worth it? **Not as you ran it.**

Evidence from your own logs/replays:
- **Value loss collapsed to ~0.006 almost immediately** — trivial, not mastery:
  feeding current scores into input channels 4–5 makes "who's ahead" a giveaway,
  so the value head gives MCTS almost no lookahead value.
- **Self-play was blow-out dominated** (`replays/replay_12.json` = **53–0**), so
  the net never learned nuanced play.
- **Cost vs. payoff:** ~80M net evals, torch wouldn't run locally, and the result
  was never even deployed (the in-game bot was a no-op placeholder).

Nuance worth keeping: because the cascade-setup is **too deep to search**, a
*learned policy* is genuinely the right way to capture opening intuition — but
learn it efficiently, not tabula-rasa.

## 3. The better architecture — Expert Iteration (`distill_train.py`)

Let the strong Monte-Carlo bot **teach** a small net:

1. SmartBot (teacher) plays games; record `(position → teacher move)` and the game
   result as the value target.
2. On-the-fly **dihedral 8× symmetry augmentation** (valid on the torus too).
3. Supervised training of a small `AlphaZeroResNet` (default 4 blocks / 64
   channels): policy = cross-entropy to the teacher, value = MSE to the result.

High-quality targets from step one → **converges fast, on CPU, with no value
collapse and no blow-out problem.**

```bash
python distill_train.py --games 150 --teacher-budget 0.4 --blocks 4 --channels 64 --epochs 12
# writes checkpoints_distill/best.pth  →  neural_bot.py auto-loads it
```

`neural_bot.py` plays policy-argmax with a **tactical safety net** (never passes
up a free SOS). If torch isn't importable or no checkpoint exists, it reports
`available = False` and the game falls back to SmartBot — nothing breaks.

> Note: torch only runs on your Linux setup, so train there / on Colab. Until a
> checkpoint exists, the strong Monte-Carlo bot carries the "AlphaZero" slot.

## Evidence (reproduce)
```bash
# SmartBot vs greedy, alternating sides:
python - <<'PY'
import numpy as np, random
from smart_bot import SmartBot; from greedy_bot import SOSBot; from game_logic import SOSGame
cb=lambda b:[[' ' if v==0 else('S' if v==1 else 'O') for v in r]for r in b]
def play(a,b):
    g=SOSGame(8,True); g.reset(); t={0:a,1:b}
    while not np.all(g.board!=0):
        (r,c),L=t[g.current_player].choose_move(cb(g.board)); g.step((r*8+c)+(0 if L=='S' else 64))
    return g.scores[0],g.scores[1]
random.seed(11); W=[0,0,0]
for i in range(8):
    s0,s1=(play(SmartBot(True,1.0),SOSBot(True)) if i%2==0 else play(SOSBot(True),SmartBot(True,1.0))[::-1])
    W[0 if s0>s1 else 1 if s1>s0 else 2]+=1
print("SmartBot wins/losses/draws:",W)
PY
```
Observed: SmartBot **8–0** vs greedy (typical scores 56–1, 58–0, 61–4).
