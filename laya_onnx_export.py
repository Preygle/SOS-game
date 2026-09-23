"""
laya_onnx_export.py  --  put Laya on an AMD GPU via ONNX Runtime + DirectML.

WHY THIS EXISTS
---------------
Laya runs on CPU here because there is no PyTorch GPU path for a Radeon on
Windows: ROCm is Linux-only, and torch-directml has no Python 3.13 build. What
does work is DirectML through ONNX Runtime, which talks to any DirectX 12 GPU --
including the RX 6700M. So the encoder is exported once to ONNX and executed by
onnxruntime-directml; laya_worker.py then swaps it in for the torch module.

Only DecisionModel.forward is replaced. Tokenisation, the option/marker layout,
temperature calibration and the answer formatting all stay in the Laya package,
so a DML run and a CPU run differ only in who multiplies the matrices -- which
is exactly what --check verifies.

    D:\\laya\\env\\Scripts\\python.exe laya_onnx_export.py --model typed-decisions
    D:\\laya\\env\\Scripts\\python.exe laya_onnx_export.py --model typed-decisions --check
"""

import argparse
import os
import sys
import time

os.environ.setdefault("HF_HOME", r"D:\laya\hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ONNX_DIR = os.environ.get("LAYA_ONNX_DIR", r"D:\laya\onnx")
MODELS = {"multilingual": "multilingual", "english": None, "typed-decisions": "typed-decisions"}


def onnx_path(model):
    return os.path.join(ONNX_DIR, "laya-%s.onnx" % model)


def sample_batch(agent, n_options=6, seq_len=256):
    """A representative batch: one question, `n_options` options, `seq_len` tokens.

    Shapes are marked dynamic at export, so these numbers only steer the tracer.
    """
    import torch
    pad = agent.tok.pad_token_id or 0
    ids = torch.randint(5, 1000, (1, seq_len), dtype=torch.long)
    att = torch.ones((1, seq_len), dtype=torch.long)
    mpos = torch.arange(1, n_options + 1, dtype=torch.long)[None, :]
    mmask = torch.ones((1, n_options), dtype=torch.bool)
    qtype = torch.zeros((1,), dtype=torch.long)
    return ids, att, mpos, mmask, qtype


def export(model_name):
    import torch
    import laya
    # The decision head is a stack of nn.TransformerEncoderLayer, which in eval
    # mode dispatches to the fused aten::_transformer_encoder_layer_fwd kernel.
    # That fused op has no ONNX translation, so turn the fast path off and let
    # the tracer see the ordinary attention/MLP modules instead.
    torch.backends.mha.set_fastpath_enabled(False)
    os.makedirs(ONNX_DIR, exist_ok=True)
    print("loading %s on CPU..." % model_name)
    agent = laya.load("convaiinnovations/laya", subfolder=MODELS[model_name], device="cpu")
    agent.model.eval()

    args = sample_batch(agent)
    out = onnx_path(model_name)
    print("exporting to %s ..." % out)
    with torch.no_grad():
        torch.onnx.export(
            agent.model, args, out,
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            output_names=["logits", "act"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "attention_mask": {0: "batch", 1: "seq"},
                          "marker_pos": {0: "batch", 1: "opts"},
                          "marker_mask": {0: "batch", 1: "opts"},
                          "qtype": {0: "batch"},
                          "logits": {0: "batch", 1: "opts"},
                          "act": {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False)
    print("exported: %.0f MB" % (os.path.getsize(out) / 1e6))
    return agent


def make_session(model_name, provider="dml"):
    import onnxruntime as ort
    providers = (["DmlExecutionProvider", "CPUExecutionProvider"] if provider == "dml"
                 else ["CPUExecutionProvider"])
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path(model_name), so, providers=providers)
    return sess


class OnnxModel:
    """Drop-in for DecisionModel: same call signature, same two outputs.

    Laya's Agent only ever calls `self.model(...)`, so replacing this attribute
    moves the whole forward pass onto the GPU without touching the library.
    """

    def __init__(self, model_name, provider="dml"):
        import torch
        self.torch = torch
        self.sess = make_session(model_name, provider)
        self.provider = self.sess.get_providers()[0]

    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
                 detach_encoder=False):
        np_in = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
            "marker_pos": marker_pos.cpu().numpy(),
            "marker_mask": marker_mask.cpu().numpy(),
            "qtype": qtype.cpu().numpy(),
        }
        logits, act = self.sess.run(["logits", "act"], np_in)
        return self.torch.from_numpy(logits), self.torch.from_numpy(act)

    # The Agent touches these during setup; after the swap they are no-ops.
    def to(self, *a, **k):
        return self

    def eval(self):
        return self


def check(model_name, agent=None):
    """Same questions through torch-CPU and ONNX-DML; compare answers and speed."""
    import numpy as np
    import laya
    if agent is None:
        agent = laya.load("convaiinnovations/laya", subfolder=MODELS[model_name], device="cpu")

    state = ("You are playing SOS.\nCURRENT GAME.\nBoard:\n"
             "8 ..S.....\n7 ...O....\n6 ..S..S..\n5 ........\n"
             "4 ....O...\n3 .S......\n2 ........\n1 .......S\n  ABCDEFGH\n\n"
             "Your score 3, opponent score 2. 41 empty cells left.")
    questions = {"move": {"type": "choice",
                          "instructions": "Pick the best plan.",
                          "criteria": {"take_points": "score 2 points now and keep the turn",
                                       "play_safe": "score nothing, give the opponent nothing",
                                       "wait": "place far from the action",
                                       "open_up": "let the opponent score 1"}}}

    def run(tag, n=10):
        for _ in range(3):
            agent.system_one(state, questions)
        t = time.perf_counter()
        for _ in range(n):
            r = agent.system_one(state, questions)
        ms = (time.perf_counter() - t) / n * 1000
        a = r["answers"]["move"]
        probs = np.array([a["probabilities"][k] for k in sorted(a["probabilities"])])
        print("  %-26s %7.0f ms   choice=%-12s" % (tag, ms, a["choice"]))
        return probs

    torch_probs = run("torch CPU")
    torch_model = agent.model
    agent.model = OnnxModel(model_name, "cpu")
    onnx_cpu_probs = run("onnx CPU")
    agent.model = OnnxModel(model_name, "dml")
    dml = agent.model.provider
    dml_probs = run("onnx %s" % dml)
    agent.model = torch_model

    for tag, p in (("onnx CPU", onnx_cpu_probs), ("onnx DML", dml_probs)):
        diff = float(np.max(np.abs(p - torch_probs)))
        verdict = "OK" if diff < 2e-3 else "DIFFERS"
        print("  max probability difference vs torch, %-9s %.2e  %s" % (tag, diff, verdict))
    return dml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="typed-decisions", choices=list(MODELS))
    p.add_argument("--check", action="store_true", help="compare CPU and GPU after export")
    args = p.parse_args()

    agent = None
    if not os.path.exists(onnx_path(args.model)):
        agent = export(args.model)
    else:
        print("already exported: %s" % onnx_path(args.model))
    if args.check:
        check(args.model, agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
