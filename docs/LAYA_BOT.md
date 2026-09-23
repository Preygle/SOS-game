# The LAYA Bot

An SOS opponent driven by **Laya**, an open-weight "System One" decision model.
This document is the architecture *and* the measurements, including the ones that
did not work out, because the negative results here are the useful part.

Files: [`laya_bot.py`](../laya_bot.py) (the bot) ·
[`laya_worker.py`](../laya_worker.py) (process bridge) ·
[`laya_probe.py`](../laya_probe.py) (does it pick the right move?) ·
[`mc_bot.py`](../mc_bot.py) (the Monte-Carlo variant) ·
[`laya_onnx_export.py`](../laya_onnx_export.py) (GPU attempt)

---

## 1. What Laya is

Laya (Convai Innovations, Apache 2.0) is a **non-autoregressive decision model**.
You give it a *state* and *typed questions*; it returns typed answers with
calibrated probabilities in a single forward pass. It never generates text, so
there is nothing to parse and nothing to hallucinate.

Three checkpoints ship in one repo:

| checkpoint | backbone | params | context | built for |
| :-- | :-- | --: | --: | :-- |
| `laya` | ModernBERT-large | 421M | 512 | English triage, guardrails |
| `laya-multilingual` | mmBERT-base | 322M | 1024 | 100+ languages, fastest |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | typed-decision workflows |

Question types are `choice` (pick a label), `score` (ordinal level) and `noul`
(yes/no).

**Measured on this machine** (Ryzen 9 5900HX, 8 threads, CPU-only torch; median
of 20 runs, warm-up excluded):

| checkpoint | 1 question | 5 questions | per question batched |
| :-- | --: | --: | --: |
| `laya-multilingual` | **128 ms** | **529 ms** | **106 ms** |
| `laya` | 402 ms | 1561 ms | 312 ms |
| `laya-typed-decisions` | 410 ms | 1604 ms | 321 ms |

Loading a checkpoint costs 21–30 s, once per process.

## 2. Why there is a separate process

Laya needs `transformers` 5.x; this project pins 4.50. Rather than upgrade the
game's ML stack for a side experiment, Laya lives in its own environment
(`D:\laya\env`) and `laya_worker.py` bridges to it over stdin/stdout, one JSON
object per line.

```
laya_bot.py  ──spawn──>  D:\laya\env\python laya_worker.py --model <ckpt>
             ──stdin──>  {"state": ..., "questions": {...}}
             <─stdout──  {"ok": true, "answers": {...}, "ms": 131.4}
```

The worker is long-lived: a decision costs milliseconds but a load costs 30
seconds, so the process is started once and kept for the whole game.

**This must never be built on the main thread.** pyglet draws from the main
thread, so a 40 s blocking load stops the window repainting and Windows reports
the game as "Not Responding" — indistinguishable from a crash. It was exactly
that at one point:

| | worst frame gap |
| :-- | --: |
| built in the clock callback | **42.5 s** |
| built on a worker thread, preloaded on menu selection | **0.03 s** |

`sos.py` now preloads on menu selection and resolves the engine inside the bot's
worker thread.

## 3. The action-space problem

A `choice` question degrades badly past ~20 labels — the model card reports 0.425
on a 77-label task, and all options share one token budget (`head_max_len`), so
more labels means fewer tokens describing each. SOS offers up to **128 moves**
(64 cells × 2 letters). Handing Laya the raw move list asks it to do the thing it
is worst at.

So the engine shortlists and Laya chooses. Three modes were built and measured:

- **`move`** — up to 8 concrete moves, each annotated with points now, chain
  size, and what the opponent scores in reply.
- **`strategy`** — four *named plans* (`take_points`, `play_safe`, `wait`,
  `open_up`) that the engine then executes. Semantic labels, which is the shape
  Laya was trained on.
- **`search`** *(default)* — alpha-beta ranks every root move, and Laya chooses
  only among moves the search rates **identically**.

## 4. Does Laya pick the right move?

`laya_probe.py` builds positions where **one shortlisted option scores a free
chain and every other option scores nothing**, then asks. A model that
understands the options should be near 100%.

**`move` mode**, 20 positions:

| checkpoint | picks the free chain | chance |
| :-- | --: | --: |
| `laya-multilingual` | 15% | 12% |
| `laya` | 5% | 12% |
| `laya-typed-decisions` | 35–45% | 12% |

**`strategy` mode**, 20 positions: `laya-typed-decisions` **60%** against a 29%
chance baseline; `laya-multilingual` 10%, below chance.

Only `typed-decisions` beats chance, and even it walks past free points four
times in ten. Two things that did *not* help:

- **Explaining the rules in depth.** Putting the full SOS rules in the state
  measured no better than leaving them out (15% vs 15%; 35% vs 45% the other
  way) and cost 2–3× the latency. The rules text is still in `laya_bot.py` and
  is off by default.
- **A bigger, better checkpoint.** The English 421M model scored *below* chance.

## 5. Playing strength

`arena.py`, alternating who moves first.

**Unaided Laya** (`strategy` mode) against the identical shortlist with the model
removed — the ablation that isolates what Laya contributes:

| matchup | record | points |
| :-- | :-- | :-- |
| Laya vs its own shortlist, first-listed plan | 0-12-0 (0%) | **0.0 – 67.5** |
| Laya vs `greedy` | 0-12-0 (0%) | 0.8 – 58.2 |

It scored zero points in twelve games. The shortlist plays better without it.

**Search-backed** (`search` mode, the default), where Laya may only choose among
moves the search cannot separate:

| matchup | record | points |
| :-- | :-- | :-- |
| vs `strong:1.0` | 2-8-0 (20%) | 6.0 – 10.8 |

From 0.0 points to competitive games. It still loses, which is itself a finding:
even picking among moves the search rates *equal* costs strength, because the
search's own move ordering carries information its value does not.

## 6. The "simulate every cell, then ask Laya" design

A natural proposal: score every move by Monte-Carlo playout, then hand the
statistics to Laya to pick. Built in `mc_bot.py` so it could be measured. The two
choosers share the identical simulation, so the match isolates one thing.

| chooser | record | points |
| :-- | :-- | :-- |
| Laya reads the simulated scores | 2-8-0 (20%) | 10.8 |
| `argmax` over the same scores | 8-2-0 (80%) | **53.2** |

Routing the numbers through Laya costs ~42 points a game. If the simulation
already produces a per-move score, `max()` beats asking a classifier to read it.

The simulation half is not strong enough on its own either:

| matchup | record | points |
| :-- | :-- | :-- |
| Monte Carlo vs the search engine | 0-12-0 (0%) | 0.0 – 74.8 |
| Monte Carlo vs `greedy` | 1-10-1 (12.5%) | 6.6 – 49.5 |

At a 1 s budget each move gets ~27 playouts, and a mean over 27 randomised
samples is mostly noise. The randomisation that makes samples differ also makes
them wrong: a random move gifts a chain, so the estimate partly measures
blunders. The main engine uses playouts too, but as the *leaf of an alpha-beta
search* with a deterministic policy — which is why it wins 12-0.

## 7. GPU: unfinished

Laya runs on CPU here. There is no PyTorch GPU path for a Radeon on Windows —
ROCm is Linux-only and `torch-directml` has no Python 3.13 build. The viable
route is ONNX Runtime with DirectML, and `DmlExecutionProvider` is present on
this machine.

`laya_onnx_export.py` exports the model (1.6 GB) after working around PyTorch's
fused attention kernel, which has no ONNX translation
(`torch.backends.mha.set_fastpath_enabled(False)`). It then **fails at runtime**:

```
Reshape node '/layers.0/self_attn/Reshape_4'
Input shape:{129,1,1024}, requested shape:{256,16,64}
```

The decision head's attention baked in the 256-token trace length despite the
dynamic axes, so it only accepts that exact sequence length. Needs re-exporting
with genuinely dynamic shapes.

Worth stating: the GPU would make this bot **faster, not better**. Nothing in
section 5 is limited by latency.

## 8. Running it

In the game, pick **LAYA** in Bot Settings, or put it in either seat of **Bot vs
Bot**. `start_game.bat` points `LAYA_PYTHON` and `HF_HOME` at the environment.

From the command line:

```bash
python laya_bot.py                  # one decision, with the shortlist printed
python laya_probe.py --n 20         # does it pick the free chain?
python arena.py laya:typed-decisions:time_budget=1.0 strong:1.0 --games 10
python mc_bot.py 2.0                # the Monte-Carlo variant
```

## 9. What this experiment actually showed

Laya is genuinely fast and genuinely calibrated, and for the jobs on its model
card — routing, triage, moderation, guardrails — the latency numbers in section 1
are excellent. Playing a combinatorial game is not one of those jobs. Every
configuration tried here measured worse than the plain search, and the two
designs that gave Laya the final say measured worse than deleting that step.

The one place it might still earn its keep is as a **move-ordering prior** for
the search, where a wrong guess costs time rather than points. That is untested.
