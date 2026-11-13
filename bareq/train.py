"""Train one Bareq run. Same architecture, loss and step budget for every run; only the data differs.

  python -m bareq.train --run all        # dark | noisy | old | all

Writes to --out (default /kaggle/working/bareq-run-<run>):
  ckpt_<run>.pt       checkpoint, saved every 15 minutes; training resumes from it automatically
  metrics_<run>.json  PSNR / SSIM on every test set, written once training is complete
  bareq_<run>.onnx    exported model (EMA weights)
  log_<run>.jsonl     loss and validation history

Resume: the newest ckpt_<run>.pt found in --out or anywhere under --input is used. On Kaggle, add the
previous version's output as an input and run again.
"""
import argparse
import copy
import json
import math
import os
import shutil
import sys
import tarfile
import time

import torch
from torch.utils.data import DataLoader

from . import data, metrics
from .model import Bareq, count_params

TOTAL_STEPS = 60000   # the fixed budget for all four runs
BATCH = 16
CROP = 256
LR = 5e-4
WARMUP = 1000
EMA = 0.999
T0 = time.time()


def log(*s):
    print(f"[{(time.time() - T0) / 60:6.1f}m]", *s, flush=True)


def find_file(root, name):
    for d, _, files in os.walk(root, followlinks=True):
        if name in files:
            return os.path.join(d, name)
    return None


def lr_at(step, total, base):
    if step < WARMUP:
        return base * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, total - WARMUP)
    return 1e-6 + (base - 1e-6) * 0.5 * (1 + math.cos(math.pi * p))


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


class Amp(torch.nn.Module):
    """Mixed precision inside forward(): DataParallel runs each GPU in its own thread, and autocast set
    outside does not reach those threads."""

    def __init__(self, m, device_type, enabled):
        super().__init__()
        self.m, self.device_type, self.enabled = m, device_type, enabled

    def forward(self, x):
        with torch.autocast(self.device_type, enabled=self.enabled):
            return self.m(x)


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def prepare_data(src, cache, run):
    """Copy this run's arrays to local disk (the Kaggle input mount is slow for random reads) and unpack val/test."""
    os.makedirs(cache, exist_ok=True)
    for f in data.RUN_FILES[run]:
        s, d = os.path.join(src, f), os.path.join(cache, f)
        if not os.path.exists(d) or os.path.getsize(d) != os.path.getsize(s):
            t = time.time()
            shutil.copyfile(s, d + ".part")
            os.replace(d + ".part", d)
            log(f"copied {f} ({os.path.getsize(d) / 1e9:.1f} GB, {time.time() - t:.0f}s)")
    for split in ("val", "test"):
        if not os.path.isdir(os.path.join(cache, split)):
            with tarfile.open(os.path.join(src, f"{split}.tar")) as t:
                t.extractall(cache)
    return cache


@torch.no_grad()
def evaluate(model, root, split, device, limit=0, tasks=("dark", "noisy", "old", "noisy_syn"), per_image=False):
    model.eval()
    out = {}
    for task in tasks:
        rows = []
        for name, pi, pt in data.list_pairs(root, split, task, limit):
            x, y = data.load_image(pi, device), data.load_image(pt, device)
            p = metrics.quantize(metrics.tiled(model, x, tile=CROP))
            rows.append(dict(name=name, psnr=float(metrics.psnr(p, y)), ssim=float(metrics.ssim(p, y)),
                             psnr_in=float(metrics.psnr(x, y)), ssim_in=float(metrics.ssim(x, y))))
        out[task] = dict(n=len(rows), **{k: sum(r[k] for r in rows) / len(rows)
                                         for k in ("psnr", "ssim", "psnr_in", "ssim_in")})
        if per_image:
            out[task]["images"] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(data.RUN_TASKS))
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default="/tmp/bareq-cache")
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-minutes", type=float, default=15)
    ap.add_argument("--max-hours", type=float, default=11.0, help="stop cleanly before the session limit")
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--val-limit", type=int, default=20)
    ap.add_argument("--eval-limit", type=int, default=0, help="smoke test: score only this many test images per task")
    ap.add_argument("--stop-after", type=int, default=0, help="smoke test: pretend the session ended at this step")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = a.out or f"/kaggle/working/bareq-run-{a.run}"
    os.makedirs(out, exist_ok=True)
    dev = torch.device(a.device)
    amp = dev.type == "cuda" and not a.no_amp
    ckpt_path = os.path.join(out, f"ckpt_{a.run}.pt")

    src = find_file(a.input, "train_clean.npy")
    if not src:
        sys.exit(f"bareq-data not found under {a.input}: attach the dataset as an input")
    root = prepare_data(os.path.dirname(src), a.cache, a.run)
    man = json.load(open(os.path.join(os.path.dirname(src), "manifest.json")))

    torch.manual_seed(a.seed)
    model = Bareq().to(dev)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.9), weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=amp)
    step, history, train_seconds = 0, [], 0.0

    # resume from the newest checkpoint for this run
    cands = [ckpt_path] if os.path.exists(ckpt_path) else []
    for d, _, files in os.walk(a.input, followlinks=True):
        cands += [os.path.join(d, f) for f in files if f == f"ckpt_{a.run}.pt"]
    best = None
    for p in cands:
        c = torch.load(p, map_location="cpu", weights_only=False)
        if best is None or c["step"] > best["step"]:
            best, best_path = c, p
    if best:
        same = (best["run"], best["total_steps"], best["batch"], best["seed"]) == (a.run, a.total_steps, a.batch, a.seed)
        if not same:
            sys.exit(f"checkpoint {best_path} was made with different settings: "
                     f"{best['run']}, {best['total_steps']} steps, batch {best['batch']}, seed {best['seed']}")
        model.load_state_dict(best["model"]), ema.load_state_dict(best["ema"])
        opt.load_state_dict(best["opt"]), scaler.load_state_dict(best["scaler"])
        step, history, train_seconds = best["step"], best["history"], best["train_seconds"]
        log(f"RESUMED from step {step} ({best_path})")
    else:
        log("starting from scratch")

    gpus = torch.cuda.device_count() if dev.type == "cuda" else 0
    net = Amp(model, dev.type, amp)
    if gpus > 1:
        net = torch.nn.DataParallel(net)
    log(f"run={a.run} steps={a.total_steps} batch={a.batch} crop={CROP} params={count_params(model) / 1e6:.2f}M "
        f"device={dev} gpus={gpus} amp={amp} torch={torch.__version__} data={man['git_commit'][:7]}")

    def save(reason):
        save_atomic(dict(run=a.run, step=step, total_steps=a.total_steps, batch=a.batch, seed=a.seed,
                         config=model.config, model=model.state_dict(), ema=ema.state_dict(),
                         opt=opt.state_dict(), scaler=scaler.state_dict(), history=history,
                         train_seconds=train_seconds, data_commit=man["git_commit"]), ckpt_path)
        with open(os.path.join(out, f"log_{a.run}.jsonl"), "w") as f:
            f.writelines(json.dumps(h) + "\n" for h in history)
        log(f"checkpoint saved at step {step} ({reason})")

    if step < a.total_steps:
        ds = data.TrainSet(root, a.run, CROP, a.seed, a.total_steps * a.batch, offset=step * a.batch)
        loader = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, drop_last=True,
                            pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0,
                            multiprocessing_context="spawn" if a.workers > 0 else None,
                            prefetch_factor=4 if a.workers > 0 else None)
        last_ckpt = t_log = time.time()
        loss_sum, loss_n, bad = 0.0, 0, 0
        model.train()
        for x, y in loader:
            x = x.to(dev, non_blocking=True).float() / 255
            y = y.to(dev, non_blocking=True).float() / 255
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.total_steps, LR)
            pred = net(x)
            loss = charbonnier(pred.float(), y)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                bad += 1
                if bad > 20:
                    sys.exit("loss is not finite, stopping")
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                loss_sum, loss_n = loss_sum + loss.item(), loss_n + 1
            step += 1
            d = min(EMA, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)

            if step % 100 == 0 or step == a.total_steps:
                dt = time.time() - t_log
                train_seconds += dt
                n = 100 if step % 100 == 0 else step % 100
                rec = dict(step=step, loss=loss_sum / max(1, loss_n), lr=opt.param_groups[0]["lr"],
                           it_s=n / dt, skipped=bad)
                history.append(rec)
                if step % 500 == 0 or step == a.total_steps or step <= 300:
                    left = (a.total_steps - step) / rec["it_s"] / 3600
                    log(f"step {step}/{a.total_steps} loss {rec['loss']:.4f} lr {rec['lr']:.1e} "
                        f"{rec['it_s']:.2f} it/s, about {left:.1f} h left")
                loss_sum, loss_n = 0.0, 0
                t_log = time.time()
            if step % a.val_every == 0 or step == a.total_steps:
                v = evaluate(ema, root, "val", dev, limit=a.val_limit, tasks=("dark", "noisy", "old"))
                history.append(dict(step=step, val={k: [round(r["psnr"], 3), round(r["ssim"], 4)] for k, r in v.items()}))
                log("val PSNR/SSIM  " + "  ".join(f"{k} {r['psnr']:.2f}/{r['ssim']:.3f}" for k, r in v.items()))
                model.train()
                t_log = time.time()
            done = step >= a.total_steps
            out_of_time = (time.time() - T0) / 3600 > a.max_hours
            fake_stop = a.stop_after and step >= a.stop_after
            if done or out_of_time or fake_stop or time.time() - last_ckpt > a.ckpt_minutes * 60:
                save("finished" if done else "time limit" if out_of_time else "test stop" if fake_stop else "periodic")
                last_ckpt = t_log = time.time()
            if done:
                break
            if out_of_time or fake_stop:
                log(f"STOPPED at step {step}/{a.total_steps}. Run again with this output attached as an input to resume.")
                return
        del loader

    # ----- finished: score on the test sets and export -----
    log("training complete, scoring test sets")
    res = evaluate(ema, root, "test", dev, limit=a.eval_limit, per_image=True)
    for k, r in res.items():
        log(f"test {k:9} n={r['n']:3}  PSNR {r['psnr']:.2f} (input {r['psnr_in']:.2f})  SSIM {r['ssim']:.4f} (input {r['ssim_in']:.4f})")
    onnx_path = os.path.join(out, f"bareq_{a.run}.onnx")
    info = {}
    try:
        from .export import export
        info = export(ema.cpu(), onnx_path)
        log(f"exported {onnx_path} ({info['file_bytes'] / 1e6:.1f} MB, max diff vs PyTorch {info['max_abs_diff_vs_torch']:.1e})")
    except Exception as e:  # the checkpoint is enough to export later, so don't lose the metrics over this
        log(f"ONNX export failed ({type(e).__name__}: {e}); export later from the checkpoint")
    its = [h["it_s"] for h in history if "it_s" in h]
    with open(os.path.join(out, f"metrics_{a.run}.json"), "w") as f:
        json.dump(dict(run=a.run, steps=step, batch=a.batch, crop=CROP, lr=LR, loss="charbonnier", seed=a.seed,
                       params=count_params(model), config=model.config, onnx=info, eval_limit=a.eval_limit,
                       train_hours=round(train_seconds / 3600, 2), mean_it_s=round(sum(its) / max(1, len(its)), 2),
                       gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", gpus=gpus, amp=amp,
                       torch=torch.__version__, data_commit=man["git_commit"],
                       code_commit=os.environ.get("BAREQ_COMMIT", "unknown"),
                       tile=CROP, overlap=32, test=res), f, indent=1)
    log("DONE")
    for f in sorted(os.listdir(out)):
        print(f"  {os.path.getsize(os.path.join(out, f)) / 1e6:8.1f} MB  {f}")


if __name__ == "__main__":
    main()
