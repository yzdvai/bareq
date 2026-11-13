"""Synthetic degradations. All functions take float32 HxWx3 RGB in [0, 1] and a numpy Generator.

- noisy(): phone-like sensor noise on a clean photo (extra data next to SIDD's real noise).
- old():   faded, yellowed, low-contrast, slightly soft and grainy, like an old print.
           No scratches, tears or missing areas: those need a generative model.
"""
import cv2
import numpy as np


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _blur(x, sigma):
    if sigma <= 0:
        return x
    return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def _lowres_noise(rng, h, w, c, scale):
    """Gaussian noise drawn at 1/scale resolution and upsampled: clumpy, like film grain or chroma blotches."""
    if scale <= 1.0:
        return rng.standard_normal((h, w, c)).astype(np.float32)
    sh, sw = max(1, int(round(h / scale))), max(1, int(round(w / scale)))
    n = rng.standard_normal((sh, sw, c)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    if n.ndim == 2:
        n = n[..., None]
    return n / (n.std() + 1e-8)


def _jpeg(x, q):
    u8 = (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def noisy_params(rng):
    return dict(
        shot=_logu(rng, 2e-4, 2e-2),      # signal-dependent variance per unit intensity
        read=_logu(rng, 2e-5, 3e-3),      # constant variance
        lum=float(rng.uniform(0.0, 0.7)),  # share of noise that is the same in R, G, B
        corr=float(rng.uniform(0.4, 1.2)) if rng.random() < 0.5 else 0.0,  # spatial correlation (demosaic/ISP)
        chroma=float(rng.uniform(0.0, 0.6)) if rng.random() < 0.4 else 0.0,  # low-frequency colour blotches
        jpeg=int(rng.integers(60, 96)) if rng.random() < 0.3 else 0,
    )


def noisy(x, rng, p=None):
    p = p or noisy_params(rng)
    h, w, _ = x.shape
    sigma = np.sqrt(p["shot"] * x + p["read"]).astype(np.float32)
    n = (1 - p["lum"]) * rng.standard_normal((h, w, 3)).astype(np.float32) \
        + p["lum"] * rng.standard_normal((h, w, 1)).astype(np.float32)
    n /= np.sqrt((1 - p["lum"]) ** 2 + p["lum"] ** 2)
    if p["corr"] > 0:
        n = _blur(n, p["corr"])
        n /= n.std() + 1e-8
    y = x + sigma * n
    if p["chroma"] > 0:
        c = _lowres_noise(rng, h, w, 3, 8.0)
        c -= c.mean(axis=2, keepdims=True)  # colour only, no brightness change
        y = y + p["chroma"] * sigma * c
    y = np.clip(y, 0, 1)
    if p["jpeg"]:
        y = _jpeg(y, p["jpeg"])
    return y.astype(np.float32), p


SEPIA = np.array([[0.393, 0.769, 0.189],
                  [0.349, 0.686, 0.168],
                  [0.272, 0.534, 0.131]], np.float32)


def old_params(rng):
    return dict(
        sat=float(rng.uniform(0.35, 1.0)),          # saturation kept
        sepia=float(rng.uniform(0.0, 0.6)),         # blend toward sepia
        cast_g=float(rng.uniform(0.9, 1.0)),        # yellow cast: green and blue pulled down
        cast_b=float(rng.uniform(0.62, 0.92)),
        lo=float(rng.uniform(0.04, 0.25)),          # lifted blacks (fade)
        hi=float(rng.uniform(0.72, 0.96)),          # dulled whites
        paper=[1.0, float(rng.uniform(0.85, 0.97)), float(rng.uniform(0.6, 0.85))],  # tint of the lifted blacks
        gamma=float(rng.uniform(0.85, 1.2)),
        blur=float(rng.uniform(0.0, 1.0)),
        vignette=float(rng.uniform(0.1, 0.35)) if rng.random() < 0.3 else 0.0,
        grain=float(rng.uniform(0.015, 0.07)),
        grain_scale=float(rng.uniform(1.0, 2.2)),
        grain_lum=float(rng.uniform(0.6, 1.0)),
    )


def old(x, rng, p=None):
    p = p or old_params(rng)
    h, w, _ = x.shape
    gray = (x @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    y = gray + p["sat"] * (x - gray)
    y = (1 - p["sepia"]) * y + p["sepia"] * np.clip(y @ SEPIA.T, 0, 1)
    y = y * np.array([1.0, p["cast_g"], p["cast_b"]], np.float32)
    y = np.clip(y, 0, 1) ** p["gamma"]
    paper = np.array(p["paper"], np.float32)
    y = p["lo"] * paper + (p["hi"] - p["lo"]) * y
    y = _blur(y, p["blur"])
    if p["vignette"] > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        y = y * (1 - p["vignette"] * np.clip(r2 / 2, 0, 1))[..., None]
    g = p["grain_lum"] * _lowres_noise(rng, h, w, 1, p["grain_scale"]) \
        + (1 - p["grain_lum"]) * _lowres_noise(rng, h, w, 3, p["grain_scale"])
    g /= np.sqrt(p["grain_lum"] ** 2 + (1 - p["grain_lum"]) ** 2)
    y = y + p["grain"] * g
    return np.clip(y, 0, 1).astype(np.float32), p


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
