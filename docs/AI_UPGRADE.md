# AI Upgrade — Stronger Rule Bot + RL Verdict

This documents the two AI changes: a much stronger classical opponent, and why
the AlphaZero approach was pivoted to **Expert Iteration**. Every claim below was
verified by self-play (see "Evidence").

## TL;DR

| | Before | After |
|---|---|---|
| **Rule bot** (`greedy_bot.py`) | "Depth-2" greedy, blind to bonus-turn chains | `smart_bot.py`: **two-pass Monte-Carlo** (greedy rollouts + adversarial verification). Beats greedy **~8–0**, often by 40+ points |
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

A **two-pass Monte-Carlo** search (drop-in for `SOSBot`):

- **Pass 1** — for each candidate move, **play the game out to the end** once
  with a deterministic, strong rollout policy and score the move by the final
  point differential.
- **Rollout policy**: take the biggest available SOS (chains via bonus turns);
  otherwise the most *clustered* quiet move that does **not** hand the opponent
  an immediate SOS; if **every** clustered move is poisoned, play an isolated
  **waiting move** far from the action (the zugzwang escape). An incremental
  per-cell activity table makes rollouts ~5–7× faster than a naive scan.
- **Pass 2 (adversarial verification)** — the top `VERIFY_CAP=6` pass-1 leaders
  are re-scored by their **worst outcome** over the opponent's strongest replies
  (best scoring chains, best quiet reply, waiting move). Because the reply set
  always contains the rollout-policy reply and everything is deterministic,
  `verified(c) ≤ pass1(c)` — an **admissible early-stop**: skip verification
  once the next candidate's optimistic value can't beat the best verified value
  (never changes the outcome, only saves time). The cap matters: *uncapped*
  verification over-optimizes the pessimistic reply model and **lost 2–10** to
  the capped version — verification should audit the leaders, not run the show.
  Typical move latency <1s; `time_budget` (~9s) is a hard cap.
- Exact **toroidal** SOS scoring identical to `game_logic._check_sos`
  (property-tested: 0 mismatches over 3,840 random placements, both wrap
  modes). No torch.

Quick check (no GUI needed):
```bash
python smart_bot.py            # prints chosen move, #rollouts, #verified, value, time
```

### What did NOT work (documented so we don't repeat it)
- A full **iterative-deepening alpha-beta** (negamax + quiescence + transposition
  table) was implemented first. It **lost to greedy ~0–6**, getting blown out
  (0–67) — and *more depth made it no better*, because the deciding cascade is
  far beyond any reachable horizon. Monte-Carlo with rollouts is the right tool.
- **Noisy rollouts** (ε-random quiet moves averaged over the budget) evaluate
  *worse* than one deterministic strong rollout (3–3 vs greedy instead of 8–0).
- The rollout-policy changes **without** the verification pass went 3–5 vs the
  previous bot; verification is what carries the win (6–2).

### In-game wiring (`sos.py`)
- `greedy_bot = SmartBot(..., time_budget=2.5)` — the rule slot, now the MC bot.
- `alpha_bot  = AlphaBot(time_budget=9.0)` — neural net if trained, else MC bot.
- The search runs on a **background thread** (`bot_turn_execute` → `bot_poll`) so
  the pyglet window stays responsive instead of freezing. Worker results carry a
  **generation token**, so a stale search from a quit-and-restarted game can
  never inject a move into the new game.

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

## Evidence (measured, alternating sides each game)

| Matchup | Result | Points |
|---|---|---|
| v1 Monte-Carlo vs old greedy | **8–0** | typical 56–1, 58–0, 61–4 |
| v2 (verify) vs v1 | **6–2** | 250–151 |
| v2 without verify vs v1 | 3–5 | 163–294 (verification carries the win) |
| waiting-move rollouts vs without | **8–4** | 368–223 |
| capped (6) vs uncapped verification | **10–2** | 445–154 |
| v2 (verify) vs old greedy | **7–0–1** | 317–51 |

Correctness: `_count_sos` property-tested against `game_logic` — **0 mismatches
in 3,840 random placements** (wrap and non-wrap); all moves legal in full
self-play games; worst-case move latency ≈0.9s at a 9s cap.

Reproduce with a quick match:
```bash
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
    a,b=(play(SmartBot(True,1.0),SOSBot(True)) if i%2==0 else play(SOSBot(True),SmartBot(True,1.0))[::-1])
    W[0 if a>b else 1 if b>a else 2]+=1
print("SmartBot wins/losses/draws:",W)
PY
```
