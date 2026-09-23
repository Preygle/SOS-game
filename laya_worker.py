"""
laya_worker.py  --  long-lived Laya process, spoken to over stdin/stdout.

Laya needs transformers 5.x while this project pins 4.50, so it lives in its own
environment (D:\\laya\\env) and this script is the bridge. laya_bot.py starts one
of these, keeps it alive for the whole game, and sends it one JSON line per move.
That matters because loading a checkpoint costs 20-30 s while a decision costs
~130 ms -- paying the load once per game instead of once per move is the whole
difference between usable and not.

Protocol, one JSON object per line in each direction:
    in : {"state": <str|dict>, "questions": {...}}          -> a decision request
       : {"cmd": "ping"}                                    -> readiness check
    out: {"ok": true, "answers": {...}, "ms": 131.4}
       : {"ok": false, "error": "..."}

Run it yourself to check the environment:
    D:\\laya\\env\\Scripts\\python.exe laya_worker.py --selftest
"""

import json
import os
import sys
import time

# Keep the weights and the cache on D:, and never reach for the network: the
# checkpoints are already downloaded and verified, and a stalled HF request
# would otherwise hang a move indefinitely.
os.environ.setdefault("HF_HOME", r"D:\laya\hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODELS = {
    "multilingual": ("convaiinnovations/laya", "multilingual"),
    "english": ("convaiinnovations/laya", None),
    "typed-decisions": ("convaiinnovations/laya", "typed-decisions"),
}


def main():
    model = "multilingual"
    selftest = "--selftest" in sys.argv
    for i, a in enumerate(sys.argv):
        if a == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
    if model not in MODELS:
        print(json.dumps({"ok": False, "error": "unknown model %r" % model}), flush=True)
        return 2

    import laya
    repo, sub = MODELS[model]
    t0 = time.perf_counter()
    agent = laya.load(repo, subfolder=sub, device="cpu")
    # The handshake line tells laya_bot.py the model is resident and answering.
    print(json.dumps({"ok": True, "ready": True, "model": model,
                      "load_s": round(time.perf_counter() - t0, 1)}), flush=True)

    if selftest:
        q = {"intent": {"type": "choice",
                        "instructions": "What does the customer want?",
                        "criteria": {"refund": "money back", "help": "technical help"}}}
        r = agent.system_one("I was charged twice, please refund me.", q)
        print(json.dumps({"ok": True, "selftest": r["answers"]["intent"]["choice"]}), flush=True)
        return 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "ping":
                print(json.dumps({"ok": True, "pong": True}), flush=True)
                continue
            if req.get("cmd") == "quit":
                return 0
            t = time.perf_counter()
            out = agent.system_one(req["state"], req["questions"])
            print(json.dumps({"ok": True, "answers": out["answers"],
                              "ms": round((time.perf_counter() - t) * 1000, 1)}), flush=True)
        except Exception as e:                      # never die on one bad move
            print(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
