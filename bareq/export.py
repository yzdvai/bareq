"""ONNX export, checked against PyTorch with onnxruntime."""
import os

import numpy as np
import torch

from .model import count_params


def conv_macs(model, h, w):
    """Multiply-accumulates of all convs for one h x w input (ignores cheap elementwise ops)."""
    total = 0

    def hook(m, i, o):
        nonlocal total
        total += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        model(torch.zeros(1, 3, h, w))
    for x in hs:
        x.remove()
    return total


def export(model, path, tile=256, opset=17, write_ref=False):
    import onnx
    import onnxruntime as ort

    model = model.float().eval()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    kw = dict(opset_version=opset, input_names=["input"], output_names=["output"],
              dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h", 3: "w"}})
    x = torch.rand(1, 3, tile, tile)
    try:
        torch.onnx.export(model, (x,), path, dynamo=False, **kw)
    except TypeError:  # older torch has no `dynamo` argument
        torch.onnx.export(model, (x,), path, **kw)
    m = onnx.load(path)
    onnx.checker.check_model(m)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    worst = 0.0
    g = torch.Generator().manual_seed(1)
    for s in (tile, 128):
        xi = torch.rand(2 if s == 128 else 1, 3, s, s, generator=g)
        with torch.no_grad():
            ref = model(xi).numpy()
        got = sess.run(None, {"input": xi.numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
        if write_ref and s == tile:
            base = os.path.splitext(path)[0]
            xi.numpy().astype("<f4").tofile(base + f".ref_in_{tile}.bin")
            ref.astype("<f4").tofile(base + f".ref_out_{tile}.bin")
    return dict(params=count_params(model), file_bytes=os.path.getsize(path),
                gmacs_per_256_tile=round(conv_macs(model, 256, 256) / 1e9, 2), opset=opset,
                ops=sorted({n.op_type for n in m.graph.node}), config=model.config,
                max_abs_diff_vs_torch=worst)
