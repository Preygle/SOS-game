"""Cross-check sos_engine against game_logic.SOSGame on random playouts."""
import random
import numpy as np
from game_logic import SOSGame
import sos_engine as E


def snapshot(p):
    return (list(p.board), list(p.filled), list(p.dead), list(p.tcount),
            set(p.threats), sorted(p.empty), p.hash, list(p.touch), set(p.hot))


def run(wrap, games=40, seed=7):
    rng = random.Random(seed)
    for g in range(games):
        game = SOSGame(8, wrap)
        game.reset()
        pos = E.Position(wrap)
        while not np.all(game.board != 0):
            legal = game.get_valid_actions()
            a = rng.choice(legal)

            # engine's predicted points must equal the rules engine's
            pred = pos.points(a)

            # make/unmake must be a perfect round trip
            before = snapshot(pos)
            gained, tok = pos.make(a)
            pos.unmake(a, tok)
            assert snapshot(pos) == before, f"unmake mismatch g{g} a{a}"

            gained, tok = pos.make(a)
            _, reward, _, _ = game.step(a)
            actual = int(reward) if reward > 0 else 0
            assert pred == actual == gained, f"score mismatch {pred} {actual} {gained}"

            # threat set must equal a brute-force recomputation
            brute = {}
            for idx in pos.empty:
                for act in (idx, 64 + idx):
                    n = pos.points(act)
                    if n:
                        brute[act] = n
            fresh = E.Position(wrap)
            fresh.set_from_chars([[' ' if v == 0 else ('S' if v == 1 else 'O')
                                   for v in row] for row in game.board])
            assert set(brute) == pos.threats == fresh.threats, "threat set mismatch"
            assert [fresh.tcount[a2] for a2 in sorted(fresh.threats)] == \
                   [pos.tcount[a2] for a2 in sorted(pos.threats)], "tcount mismatch"
            assert fresh.hash == pos.hash, "hash mismatch"
            assert fresh.touch == pos.touch, "touch mismatch"
            brute_touch = [0] * 64
            for l, (x, y, z) in enumerate(pos.lines):
                if pos.filled[l] and not pos.dead[l]:
                    brute_touch[x] += 1
                    brute_touch[y] += 1
                    brute_touch[z] += 1
            assert brute_touch == pos.touch, "touch != brute force"
            assert pos.hot == {i for i in pos.empty if pos.touch[i]},                 "hot set out of step with touch"
            assert fresh.hot == pos.hot, "hot mismatch"
        assert sum(game.scores.values()) == len(game.sos_patterns)
    return True


if __name__ == "__main__":
    for wrap in (True, False):
        run(wrap)
        print(f"wrap={wrap}: engine matches game_logic on 40 random playouts  OK")
