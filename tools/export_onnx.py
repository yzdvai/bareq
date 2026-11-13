"""Export a Bareq model to ONNX and check it against PyTorch with onnxruntime.

  python tools/export_onnx.py                       # untrained model (day-one browser test)
  python tools/export_onnx.py --ckpt best.pt        # trained weights
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bareq.model import Bareq, count_params  # noqa: E402


def conv_macs(model, h, w):
    """Multiply-accumulates of all convs for one h x w input (ignores cheap elementwise ops)."""
    total = 0

    def hook(m, i, o):
        nonlocal total
        k = m.kernel_size[0] * m.kernel_size[1]
        total += o.numel() * (m.in_channels // m.groups) * k

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        model(torch.zeros(1, 3, h, w))
    for x in hs:
        x.remove()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="export/bareq.onnx")
    ap.add_argument("--tile", type=int, default=256, help="shape used for tracing and checks")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu")
        model = Bareq(**ck["config"])
        model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    else:
        model = Bareq()
        # Untrained NAFNet starts as identity (beta/gamma = 0). Randomise them so the
        # export check exercises every branch instead of a no-op graph.
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith("beta") or n.endswith("gamma"):
                    p.normal_(0, 0.1)
    model.eval()

    x = torch.rand(1, 3, a.tile, a.tile)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.onnx.export(
        model, (x,), a.out, opset_version=a.opset, dynamo=False,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h", 3: "w"}},
    )

    import onnx
    import onnxruntime as ort

    m = onnx.load(a.out)
    onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    worst = 0.0
    for s in (a.tile, 128, 512):
        xi = torch.rand(1, 3, s, s)
        with torch.no_grad():
            ref = model(xi).numpy()
        t = time.time()
        got = sess.run(None, {"input": xi.numpy()})[0]
        dt = time.time() - t
        err = float(np.abs(ref - got).max())
        worst = max(worst, err)
        print(f"{s}x{s}: max abs diff {err:.2e}, onnxruntime cpu {dt*1000:.0f} ms")

    # reference pair for the browser check (raw little-endian float32, NCHW)
    g = torch.Generator().manual_seed(1)
    xi = torch.rand(1, 3, a.tile, a.tile, generator=g)
    with torch.no_grad():
        yo = model(xi)
    base = os.path.splitext(a.out)[0]
    xi.numpy().astype("<f4").tofile(base + f".ref_in_{a.tile}.bin")
    yo.numpy().astype("<f4").tofile(base + f".ref_out_{a.tile}.bin")

    info = {
        "params": count_params(model),
        "file_bytes": os.path.getsize(a.out),
        "gmacs_per_256_tile": round(conv_macs(model, 256, 256) / 1e9, 2),
        "opset": a.opset,
        "ops": ops,
        "config": model.config,
        "max_abs_diff_vs_torch": worst,
    }
    print(json.dumps(info, indent=1))
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()
