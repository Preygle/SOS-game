# SOS AI Architecture

The game's opponent is a **search engine**, not a neural network. This document
says what it does, why the neural approach was benched, and how every claim here
was measured so you can re-run it.

Files: [`sos_engine.py`](../sos_engine.py) (rules core) ·
[`strong_bot.py`](../strong_bot.py) (search) ·
[`arena.py`](../arena.py) (measurement) ·
[`test_strong_bot.py`](../test_strong_bot.py) (correctness)

---

## 1. The one thing that makes SOS hard

Scoring an SOS grants a **bonus turn**. That single rule turns SOS into a
cascade / zugzwang game with the same shape as Dots-and-Boxes:

- Once the board is dense, the player on move can score repeatedly and sweep
  tens of points in one unbroken chain.
- So the game is not decided by who takes points early. It is decided by **who
  is forced to open the board** when safe moves run out.

Everything below follows from that. A bot that cannot see who runs out of safe
moves first is playing a different game from the one on the screen.

## 2. The rules core: 256 lines, not 64 cells

An SOS occupies three collinear cells `(a, b, c)` and is complete exactly when
`a='S'`, `b='O'`, `c='S'`. On the 8×8 torus there are `64 middles × 4 axes =
**256** such lines`, and every cell lies on **12** of them (4 as the middle, 8 as
an end). `sos_engine.py` stores the game as those 256 three-slot patterns:

| per line | meaning |
| :-- | :-- |
| `filled[l]` | slots already holding the letter that line needs |
| `dead[l]` | slots holding the wrong letter — the line can never score |

A placement touches exactly 12 lines, so make/unmake is O(12). This buys three
things for free:

- **Points for a move** — `tcount[action]`, an array lookup.
- **The threat set** — lines with `filled==2, dead==0`. Such a line has exactly
  one empty slot, so it *names* the cell and letter that scores it.
- **`touch[cell]`** — live lines through a cell that are already started. Cells
  with `touch==0` are "cold": every placement there is a safe waiting move with
  no tactical content, so the search expands **one** of them instead of ~50
  identical ones.

Throughput: **~190,000 make+unmake per second** in pure Python.

`verify_engine.py` cross-checks all of this against `game_logic.py` on random
playouts — points, threat sets, hashes, the hot set, and make/unmake round trips
— on both the toroidal and bounded boards.

## 3. The search

Negamax on score differential. The bonus turn is the only twist:

```
value(move) = points + value(child, same side to move)      # scored: keep turn
value(move) =        - value(child, other side to move)     # quiet: hand over
```

Alpha/beta pass straight through on the scoring branch (still the same
maximiser) and are negated on the quiet branch.

**Depth is counted in turns, not placements.** A quiet move costs 4 units, a
scoring move costs 1. So depth measures *hand-overs* — the unit that actually
matters — and the search follows a 20-point cascade to its end almost for free
instead of spending its whole budget inside one player's chain.

**Transpositions are keyed on the board alone.** The accumulated score is only a
constant offset and the rules depend on nothing but the letters on the board, so
the optimal future differential is a function of the position, independent of
whose turn it is. One table therefore serves both players, every move order, and
every move of the real game.

**The endgame is solved exactly.** Below ~12 empty cells the search generates
every move and runs to the end of the game; in practice it reports `solved` from
about 13 empties down.

## 4. What actually made it strong: the leaf evaluator

This was the whole ball game, and it is worth stating plainly because the
intuition points the wrong way.

The first version used a standard quiescence search: at the depth limit, keep
searching scoring moves, and value a quiet position at 0 ("assume the rest is
even"). It searched **9 turns deep** — and lost to the old `smart_bot` **4-11**.

The replacement values a leaf with a **greedy playout to the end of the game**
(`sos_engine.rollout`): take the biggest SOS and keep taking; otherwise play the
most clustered move that hands over no threat; spend a far-from-the-action
waiting move only when every clustered move is poisoned. That searches only
**3-4 turns deep** — and beats the same opponent **14-2**.

| leaf evaluator | depth reached | vs `smart:1.0`, 16 games |
| :-- | :-- | :-- |
| quiescence, quiet position = 0 | 9 turns | 4-11-1 (28%) |
| greedy playout to game end | 3-4 turns | **14-2-0 (88%)** |

Head to head at the same budget, the playout version scores **87.5%** against
the quiescence version (16 games). Nothing else tried here came close to that
swing.

The reason: with tens of points still on the board, "assume the rest is even" is
not a small error, it is a catastrophic one, and no amount of depth repairs a
leaf value that is wrong by 30 points. Depth still helps (section 5) -- it just
cannot rescue a broken leaf.

**The playout is memoised.** It is deterministic given the board, so caching it
by position hash is exact rather than approximate — and it pays twice: once for
transpositions inside a search, once for every node the next iterative-deepening
pass re-visits. This lifted midgame depth from 4-5 turns to 6-7.

## 5. Measured strength

`arena.py` runs games in parallel across cores, alternating who moves first, and
splits the record by seat.

| matchup | games | record | avg points |
| :-- | --: | :-- | :-- |
| `strong:1.0` vs `smart:1.0` | 40 | **31-9-0** (78%) | 30.3 – 7.0 |
| `strong:3.0` vs `smart:1.0` | 16 | **14-2-0** (88%) | 32.8 – 5.5 |
| `strong:1.0` vs `greedy` | 16 | **14-1-1** (91%) | 15.1 – 2.5 |
| `strong:0.8` vs `smart:2.5` (its old in-game budget) | 16 | **12-4-0** (75%) | 29.7 – 7.6 |
| `strong:0.8` vs `smart:1.0`, bounded board | 16 | **14-1-1** (91%) | 20.2 – 3.0 |
| `strong:0.8` vs `greedy`, bounded board | 16 | **14-1-1** (91%) | 6.2 – 1.3 |

Reproduce any row with e.g.

```bash
python arena.py strong:1.0 smart:1.0 --games 40
```

Sixteen games is roughly +/-12% on the score, so treat the small samples as
indicative and the 40-game row as the headline number. No seat bias: against
`greedy` the engine scores 92% moving first and 100% moving second.

**Time helps — but only self-play hides it.** Against a common opponent,
thinking longer is clearly worth it:

| | vs `smart:1.0`, 20 games |
| :-- | :-- |
| `strong:0.25` | 16-4-0 (80%), +16.4 points |
| `strong:2.5` | **18-2-0 (90%), +28.9 points** |

Head to head the same gap almost vanishes (`strong:1.5` vs `strong:0.3` is
13-10-1, 56%) — not because depth is worthless but because two engines of this
family neutralise each other into low-scoring draws. Do not tune this bot by
self-play alone; always confirm against a fixed reference opponent. That trap
cost real time here: an early "depth barely matters" reading came straight from
a self-play A/B and was simply wrong.

Move-generation width, by contrast, genuinely does not matter: 50% at double
width and 48% at half, over 40 games each. Extra breadth is wasted; extra depth
is not.

Strong-vs-strong games end low-scoring and often drawn (4-4 is typical), which
is what near-optimal play in this game looks like: both sides spend the game
destroying scoring potential rather than building it.

## 6. Correctness

`python test_strong_bot.py`

The load-bearing test is **endgame optimality**. A reference solver — which
shares no code with `sos_engine`, counting SOS with `greedy_bot`'s own scanner —
brute-forces random 6-empty endgames. The engine must report the same value
*and* play a move that preserves it. That catches sign errors, bonus-turn
mistakes and transposition-table bugs, which are precisely the failures a game
bot hides: it keeps playing, just badly.

The suite also checks it takes a double SOS over a single, keeps every move
legal for a whole game inside its clock, and rarely hands over a free SOS when a
safe move exists (3/45 in a sampled game — occasional deliberate gifts are a
real technique here, so this is a rate check, not a perfection check).

---

## 7. The neural branch, and why it is not the opponent

`models.py`, `alpha_mcts.py`, `distill_train.py` and `neural_bot.py` implement
Expert Iteration: a small ResNet distilled from the search bot's moves. It still
runs, and `distill_train.py` now uses the new engine as its teacher.

It is **not** wired into the game, because it was measured:

| NeuralBot (4×64, 40 epochs) | record | avg points |
| :-- | :-- | :-- |
| vs random | 6-0-0 | 69.0 – 0.0 |
| vs greedy | 3-3-0 | 14.8 – **23.8** |
| vs `smart:0.3` | 2-4-0 | 16.5 – **36.5** |

It could not beat the *greedy* bot — and it was sitting behind the menu entry
labelled "AlphaZero", so choosing the strongest-sounding opponent handed you the
weakest one in the project.

This is not a training-length problem. The bot plays **one forward pass with no
search**. On this machine's CPU-only torch a forward pass costs about 1 ms while
a full playout costs about 0.2 ms — so the same millisecond buys strictly more
strength spent on search than on the network. More epochs sharpen the imitation
of a teacher the student then plays without; they do not close that gap.

A network *would* pay off given a GPU and MCTS at play time (network-guided
search, AlphaZero-style, rather than a bare policy net). On a CPU-only box,
search wins. If you want a stronger opponent, raise the time budget in
`sos.py`; if you want to study distillation, `run_full_training.bat` still works
and now learns from a better teacher.
