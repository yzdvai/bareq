"""PSNR, SSIM and tiled inference. Images are float tensors N x 3 x H x W in [0, 1]."""
import torch
import torch.nn.functional as F


def psnr(a, b):
    mse = ((a - b) ** 2).mean((1, 2, 3)).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def _window(device, dtype, size=11, sigma=1.5):
    g = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim(a, b):
    """Standard SSIM (Wang et al. 2004): 11x11 Gaussian window, sigma 1.5, mean over RGB channels."""
    a, b = a.double(), b.double()
    w = _window(a.device, a.dtype)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = F.conv2d(a, w, groups=3), F.conv2d(b, w, groups=3)
    va = F.conv2d(a * a, w, groups=3) - mu_a ** 2
    vb = F.conv2d(b * b, w, groups=3) - mu_b ** 2
    cov = F.conv2d(a * b, w, groups=3) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.mean((1, 2, 3))


def quantize(x):
    """Round to 8-bit, as a saved image would be."""
    return (x.clamp(0, 1) * 255).round() / 255


def tile_positions(length, tile, stride):
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile, stride))
    pos.append(length - tile)
    return pos


def blend_window(tile, overlap, device):
    """1 in the middle, linear ramp over `overlap` pixels at each edge. The browser uses the same window."""
    ramp = (torch.arange(overlap, device=device, dtype=torch.float32) + 1) / (overlap + 1)
    w = torch.ones(tile, device=device)
    w[:overlap] = ramp
    w[-overlap:] = ramp.flip(0)
    return w[:, None] * w[None, :]


@torch.no_grad()
def tiled(model, x, tile=256, overlap=32, batch=8):
    """Run `model` on one image (1 x 3 x H x W) in overlapping tiles and blend. Same scheme as the web page,
    and the tile size equals the training crop, so the model sees the statistics it was trained on."""
    _, _, h, w = x.shape
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        # reflect needs the padding to be smaller than the image; very small images repeat the edge instead
        x = F.pad(x, (0, pw, 0, ph), mode="reflect" if ph < h and pw < w else "replicate")
    H, W = x.shape[-2:]
    stride = tile - overlap
    pos = [(y, xx) for y in tile_positions(H, tile, stride) for xx in tile_positions(W, tile, stride)]
    win = blend_window(tile, overlap, x.device)
    out = torch.zeros_like(x)
    norm = torch.zeros(1, 1, H, W, device=x.device)
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        tiles = torch.cat([x[:, :, y:y + tile, xx:xx + tile] for y, xx in chunk])
        res = model(tiles).float()
        for (y, xx), r in zip(chunk, res):
            out[0, :, y:y + tile, xx:xx + tile] += r * win
            norm[0, :, y:y + tile, xx:xx + tile] += win
    return (out / norm)[:, :, :h, :w]
