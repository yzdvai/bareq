"""Final scoring, on any machine (CPU is fine). Builds the file the web page reads.

  python -m bareq.evaluate --data data --runs runs --out results

  --data   folder with the test/ and val/ image trees (unpacked test.tar and val.tar)
  --runs   folder with bareq_<run>.onnx and metrics_<run>.json from the four training runs

Writes results/results.json and results/examples/*.webp:
  - two reference rows that need no training: the untouched input, and a classical method per task
  - the four models' scores as measured on Kaggle (PyTorch), and a re-check of the exported ONNX files
    with onnxruntime, which is what the browser runs
  - for the shipped model, the two largest and two smallest PSNR gains per task as example images
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from . import baselines, data, metrics

TASKS = ("dark", "noisy", "old")
ALL_TASKS = TASKS + ("noisy_syn",)
RUNS = ("dark", "noisy", "old", "all")


def find(root, name):
    for d, _, files in os.walk(root):
        if name in files:
            return os.path.join(d, name)
    return None


def read(path):
    return np.array(Image.open(path).convert("RGB"))


def to_t(a):
    return torch.from_numpy(a).permute(2, 0, 1)[None].float() / 255


def score(pred, target):
    p, t = to_t(pred), to_t(target)
    return float(metrics.psnr(p, t)), float(metrics.ssim(p, t))


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


class OnnxModel:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, x):
        return torch.from_numpy(self.sess.run(None, {"input": x.numpy()})[0])

    def run(self, img):
        out = metrics.quantize(metrics.tiled(self, to_t(img), tile=256, overlap=32))
        return (out[0].permute(1, 2, 0) * 255).round().byte().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ship", default="all")
    ap.add_argument("--limit", type=int, default=0, help="quick test: only this many images per task")
    ap.add_argument("--check", type=int, default=12, help="images per task to re-score for the non-shipped ONNX files")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(a.out, "examples"), exist_ok=True)
    pairs = {t: data.list_pairs(a.data, "test", t, a.limit) for t in ALL_TASKS}
    res = dict(tasks=list(TASKS), extra_tasks=["noisy_syn"], n={t: len(pairs[t]) for t in ALL_TASKS})

    # ----- reference row 1: the untouched input -----
    inp = {}
    for t in ALL_TASKS:
        rows = [dict(zip(("psnr", "ssim"), score(read(pi), read(pt)))) for _, pi, pt in pairs[t]]
        inp[t] = dict(psnr=mean(rows, "psnr"), ssim=mean(rows, "ssim"))
    res["input"] = inp
    print("input     ", {t: round(v["psnr"], 2) for t, v in inp.items()}, flush=True)

    # ----- reference row 2: a classical method per task, strength picked on val -----
    cl = {}
    for t in ALL_TASKS:
        fn, grid = baselines.METHODS[t], baselines.GRID[t]
        h = grid[0]
        if len(grid) > 1:
            val = data.list_pairs(a.data, "val", t, 12)
            val = [(read(pi), read(pt)) for _, pi, pt in val]
            h = max(grid, key=lambda g: sum(score(fn(x, g), y)[0] for x, y in val))
        rows = [dict(zip(("psnr", "ssim"), score(fn(read(pi), h), read(pt)))) for _, pi, pt in pairs[t]]
        cl[t] = dict(method=baselines.NAMES[t], strength=h, psnr=mean(rows, "psnr"), ssim=mean(rows, "ssim"))
    res["classical"] = cl
    print("classical ", {t: round(v["psnr"], 2) for t, v in cl.items()}, flush=True)

    # ----- the four models -----
    res["models"], ship_rows = {}, None
    for run in RUNS:
        mj, onnx = find(a.runs, f"metrics_{run}.json"), find(a.runs, f"bareq_{run}.onnx")
        if not mj or not onnx:
            print(f"{run:9} missing ({'metrics' if not mj else 'onnx'} not found), skipped", flush=True)
            continue
        m = json.load(open(mj))
        kag = {t: {r["name"]: r for r in m["test"][t]["images"]} for t in ALL_TASKS}
        model = OnnxModel(onnx)
        full = run == a.ship
        diffs, rows_by_task = [], {}
        for t in ALL_TASKS:
            sub = pairs[t] if full else data.list_pairs(a.data, "test", t, a.check)
            rows = []
            for name, pi, pt in sub:
                x, y = read(pi), read(pt)
                out = model.run(x)
                p, s = score(out, y)
                pin, sin = score(x, y)
                rows.append(dict(name=name, psnr=p, ssim=s, psnr_in=pin, ssim_in=sin, pi=pi, pt=pt))
                if name in kag[t]:
                    diffs.append(abs(p - kag[t][name]["psnr"]))
            rows_by_task[t] = rows
        if full:
            ship_rows = rows_by_task
        res["models"][run] = dict(
            test={t: dict(psnr=m["test"][t]["psnr"], ssim=m["test"][t]["ssim"], n=m["test"][t]["n"]) for t in ALL_TASKS},
            params=m["params"], onnx_bytes=os.path.getsize(onnx), steps=m["steps"], batch=m["batch"],
            train_hours=m["train_hours"], gpu=m["gpu"], gpus=m["gpus"], code_commit=m["code_commit"],
            data_commit=m["data_commit"], eval_limit=m.get("eval_limit", 0),
            onnx_check=dict(images=len(diffs), max_psnr_diff=round(max(diffs), 4) if diffs else None,
                            mean_psnr_diff=round(sum(diffs) / len(diffs), 4) if diffs else None, full=full))
        if full:  # the shipped model's table row comes from the ONNX file itself, on every test image
            res["models"][run]["onnx_test"] = {t: dict(psnr=mean(r, "psnr"), ssim=mean(r, "ssim"), n=len(r))
                                               for t, r in rows_by_task.items()}
        print(f"{run:9}", {t: round(m['test'][t]['psnr'], 2) for t in ALL_TASKS},
              "| ONNX vs PyTorch PSNR diff: max", res["models"][run]["onnx_check"]["max_psnr_diff"],
              f"on {len(diffs)} images", flush=True)

    # ----- examples from the shipped model: largest and smallest PSNR gain per task -----
    res["examples"] = []
    if ship_rows:
        model = OnnxModel(find(a.runs, f"bareq_{a.ship}.onnx"))
        for t in TASKS:
            rows = sorted(ship_rows[t], key=lambda r: r["psnr"] - r["psnr_in"])
            picks = [("fail", r) for r in rows[:2]] + [("ok", r) for r in rows[::-1][:2]]
            for k, (kind, r) in enumerate(picks):
                x, y = read(r["pi"]), read(r["pt"])
                out = model.run(x)
                h, w = x.shape[:2]
                s = min(h, w, 512)
                y0, x0 = (h - s) // 2, (w - s) // 2
                base = f"{t}_{kind}{k % 2 + 1}"
                for tag, im in (("in", x), ("out", out), ("gt", y)):
                    Image.fromarray(im[y0:y0 + s, x0:x0 + s]).save(
                        os.path.join(a.out, "examples", f"{base}_{tag}.webp"), quality=88, method=6)
                res["examples"].append(dict(task=t, kind=kind, id=base, source=r["name"], size=s,
                                            psnr_in=round(r["psnr_in"], 2), psnr_out=round(r["psnr"], 2),
                                            ssim_in=round(r["ssim_in"], 4), ssim_out=round(r["ssim"], 4)))

    res["meta"] = dict(ship=a.ship, tile=256, overlap=32, limit=a.limit,
                       metric_note="PSNR and SSIM on 8-bit RGB, averaged over images; SSIM with an 11x11 Gaussian window",
                       created=time.strftime("%Y-%m-%d"), seconds=round(time.time() - t0))
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out}/results.json and {len(res['examples']) * 3} example images in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
