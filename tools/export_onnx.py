"""Export a Bareq model to ONNX, plus a reference input/output pair for the browser check.

  python tools/export_onnx.py                       # untrained model (day-one browser test)
  python tools/export_onnx.py --ckpt ckpt_all.pt    # trained weights (EMA)
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bareq.export import export  # noqa: E402
from bareq.model import Bareq  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="export/bareq.onnx")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        model = Bareq(**ck["config"])
        model.load_state_dict(ck["ema"])
    else:
        model = Bareq()
        # An untrained NAFNet is the identity (beta/gamma = 0). Randomise them so the check uses every branch.
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith("beta") or n.endswith("gamma"):
                    p.normal_(0, 0.1)
    info = export(model, a.out, write_ref=True)
    print(json.dumps(info, indent=1))
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()
