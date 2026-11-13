"""Classical reference methods, one per task. uint8 RGB in, uint8 RGB out. No training involved.

  dark   histogram equalisation of the brightness channel
  noisy  non-local means denoising (OpenCV), strength picked on the validation set
  old    auto levels per colour channel (stretches contrast and removes the colour cast),
         then non-local means for the grain, strength picked on the validation set
"""
import cv2
import numpy as np

NAMES = {"dark": "Histogram equalisation", "noisy": "Non-local means",
         "old": "Auto levels + non-local means", "noisy_syn": "Non-local means"}
GRID = {"dark": [0], "noisy": [3, 5, 8, 12, 16], "old": [0, 4, 8, 12, 16], "noisy_syn": [3, 5, 8, 12, 16]}


def _nlm(x, h):
    if h <= 0:
        return x
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21), cv2.COLOR_BGR2RGB)


def dark(x, h=0):
    ycc = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def noisy(x, h=8):
    return _nlm(x, h)


def old(x, h=8):
    f = x.astype(np.float32)
    lo, hi = np.percentile(f, 1, axis=(0, 1)), np.percentile(f, 99, axis=(0, 1))
    f = (f - lo) / np.maximum(hi - lo, 1e-3) * 255
    return _nlm(np.clip(f + 0.5, 0, 255).astype(np.uint8), h)


METHODS = {"dark": dark, "noisy": noisy, "old": old, "noisy_syn": noisy}
