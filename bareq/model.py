"""Bareq: a small NAFNet-style restoration network.

Based on NAFNet (Chen et al., "Simple Baselines for Image Restoration", ECCV 2022),
shrunk to a few million parameters so it can run in a browser.
Only plain ops (conv, mean, sqrt, mul, depth-to-space) so it exports cleanly to ONNX.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an NCHW tensor, written with basic ops."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.eps = eps

    def forward(self, x):
        if x.dtype != torch.float32:  # mixed precision: keep the statistics in float32 (squares overflow in float16)
            x = x.float()
        mu = x.mean(1, keepdim=True)
        d = x - mu
        var = (d * d).mean(1, keepdim=True)
        return d / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    """Split channels in half and multiply. The half size is fixed so ONNX gets a constant Split."""

    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        a, b = x.split([self.c, self.c], dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, expand=2, ffn_expand=2):
        super().__init__()
        dw = c * expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate(dw // 2)
        self.sg2 = SimpleGate(c * ffn_expand // 2)
        # simplified channel attention: global average, then 1x1 conv
        self.sca = nn.Conv2d(dw // 2, dw // 2, 1)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.norm2 = LayerNorm2d(c)
        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.sg(self.conv2(self.conv1(self.norm1(inp))))
        x = x * self.sca(x.mean((2, 3), keepdim=True))
        y = inp + self.conv3(x) * self.beta
        x = self.conv5(self.sg2(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class Bareq(nn.Module):
    def __init__(self, width=24, enc=(2, 2, 2, 4), middle=2, dec=(2, 2, 2, 2)):
        super().__init__()
        self.config = dict(width=width, enc=list(enc), middle=middle, dec=list(dec))
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, stride=2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.pad_to = 2 ** len(enc)

    def forward(self, inp):
        # inp: N x 3 x H x W in [0, 1]; H and W must be multiples of self.pad_to
        x = self.intro(inp)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = dec(up(x) + skip)
        return self.ending(x) + inp

    def forward_any(self, inp):
        """Pad to a multiple of pad_to, run, crop back. For eval in PyTorch."""
        h, w = inp.shape[-2:]
        p = self.pad_to
        ph, pw = (p - h % p) % p, (p - w % p) % p
        x = F.pad(inp, (0, pw, 0, ph), mode="reflect")
        return self(x)[..., :h, :w]


def count_params(m):
    return sum(p.numel() for p in m.parameters())
