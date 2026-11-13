"""Time one training step under different settings, on random data. Run on the training GPU.

  python -m bareq.bench
"""
import time

import torch

from . import model as M
from .train import Amp, charbonnier


def run(name, gpus, amp, cl, bench, native, batch=16, crop=256, warm=4, steps=12):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = bench
    M.LayerNorm2d.native = native
    fmt = torch.channels_last if cl else torch.contiguous_format
    net = M.Bareq().to(dev).to(memory_format=fmt)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=amp)
    f = Amp(net, dev.type, amp)
    if gpus > 1:
        f = torch.nn.DataParallel(f)
    x = torch.rand(batch, 3, crop, crop, device=dev).contiguous(memory_format=fmt)
    y = torch.rand(batch, 3, crop, crop, device=dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(warm + steps):
        if i == warm:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
        loss = charbonnier(f(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        loss.item()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t) / steps
    mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0
    print(f"{name:42} {1 / dt:6.2f} it/s   {dt * 1000:6.0f} ms/step   {mem:4.1f} GB on GPU 0", flush=True)


def main():
    n = torch.cuda.device_count()
    print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(n)], flush=True)
    if n == 0:
        run("cpu, tiny", 1, False, False, False, False, batch=1, crop=64, warm=1, steps=2)
        run("cpu, tiny, built-in norm", 1, False, False, False, True, batch=1, crop=64, warm=1, steps=2)
        return
    #    name                                     gpus amp    cl     bench  native
    run("1 GPU, mixed precision", 1, True, False, False, False)
    run("1 GPU, mixed precision, cudnn.benchmark", 1, True, False, True, False)
    run("1 GPU, full precision", 1, False, False, False, False)
    run("1 GPU, mixed, built-in norm", 1, True, False, False, True)
    run("1 GPU, mixed, built-in norm, channels_last", 1, True, True, False, True)
    run("1 GPU, full precision, built-in norm", 1, False, False, False, True)
    if n > 1:
        run("2 GPUs, mixed precision (smoke test setup)", 2, True, False, True, False)
        run("2 GPUs, mixed precision, no benchmark", 2, True, False, False, False)
        run("2 GPUs, mixed, built-in norm", 2, True, False, False, True)
        run("2 GPUs, mixed, built-in norm, batch 32", 2, True, False, False, True, batch=32)
    run("1 GPU, mixed, built-in norm, batch 8", 1, True, False, False, True, batch=8)


if __name__ == "__main__":
    main()
