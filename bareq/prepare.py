"""Build the Bareq dataset: split first, then crop and degrade.

  python -m bareq.prepare --input /kaggle/input --out /kaggle/working/bareq-data

Written to --out:
  train_dark_low.npy, train_dark_high.npy   LOL v1 + v2-Real pairs, full size (random crops at train time)
  train_sidd_noisy.npy, train_sidd_gt.npy   SIDD Small sRGB, 512px patches, train scenes only
  train_clean.npy                           DIV2K 512px patches (noisy/old inputs are made on the fly in training)
  val.tar, test.tar                         PNG trees: {dark,noisy,old,noisy_syn}/{input,target}/, fixed seeds
  manifest.json                             every split, the LOL duplicate report, seeds, counts, params
  dedupe_check.jpg                          LOL train/test pairs either side of the duplicate threshold
  preview.jpg                               one test pair per task

Training arrays are uint8 N x H x W x 3, so training reads random crops without decoding images.

Leakage rules:
  - LOL: v1 and v2-Real are pooled. Every official test image stays in the test set, except exact copies of
    another test image. Any train image whose normal-light photo is close to ANY test image (same scene,
    or a copy of it from the other version) is dropped. Copies inside train are dropped too.
  - SIDD: split by scene number, so a scene is never in two splits.
  - DIV2K: one split shared by synthetic noise and "old"; the official valid set (0801-0900) is the test set.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

from . import degrade

SEED = 20261113
SIDD_TEST_SCENES = ["003", "005"]   # all five phones, ISO up to 10000
SIDD_VAL_SCENES = ["004"]
SIDD_TRAIN_PATCHES = 24             # per image, 512px, non-overlapping
SIDD_TEST_CROPS = 8                 # per image, 512px
SIDD_VAL_CROPS = 4
DIV2K_VAL = 40                      # taken from the 800 train images
DIV2K_FULL_PATCHES = 6              # per image at full size (plus what fits at half size)
LOL_VAL_SEEDS = 40
LOL_COPY = 0.10                     # normalised-thumbnail distance: same bright photo
LOL_LOW_COPY = 0.25                 # full-size mean abs difference of the dark images (0-255): same photo
LOL_SCENE = 0.35                    # below this, treat as the same scene (conservative)
PATCH = 512
TEST_LONG_SIDE = 1024               # DIV2K test images are resized to this before degrading
T0 = time.time()


def log(*s):
    print(f"[{time.time() - T0:6.0f}s]", *s, flush=True)


# ---------- io ----------

def imread(p):
    x = cv2.imread(p, cv2.IMREAD_COLOR)
    if x is None:
        raise IOError(p)
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


def imwrite(p, x):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if x.dtype != np.uint8:
        x = degrade.to_u8(x)
    cv2.imwrite(p, cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])


def read_many(paths, fn=imread, threads=16):
    """Kaggle's input mount has high per-file latency, so read in parallel threads."""
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(fn, paths))


def img_size(p):
    with Image.open(p) as im:
        w, h = im.size
    return h, w


def find_dir(root, name, maxdepth=6):
    """Kaggle mounts datasets at different depths; find a folder by name."""
    root_depth = root.rstrip(os.sep).count(os.sep)
    for d, subdirs, _ in os.walk(root, followlinks=True):
        if os.path.basename(d) == name:
            return d
        if d.count(os.sep) - root_depth >= maxdepth:
            subdirs[:] = []
    raise FileNotFoundError(f"no folder named {name!r} under {root}")


def pngs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))


def resize_long(x, n):
    h, w = x.shape[:2]
    s = n / max(h, w)
    if s >= 1:
        return x
    return cv2.resize(x, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def crop16(x):
    """Crop to a multiple of 16 so the model needs no padding at test time."""
    h, w = x.shape[:2]
    return x[: h - h % 16, : w - w % 16]


def grid_positions(h, w, size, n, rng):
    """Up to n non-overlapping size x size crops on a grid, chosen at random."""
    ys, xs = np.arange(0, h - size + 1, size), np.arange(0, w - size + 1, size)
    pos = [(int(y), int(x)) for y in ys for x in xs]
    idx = rng.permutation(len(pos))[:n]
    return [pos[i] for i in sorted(idx)]


def new_npy(path, n, shape):
    a = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(n,) + tuple(shape))
    del a


def progress(it, total, what, every=0.1):
    step, nxt = max(1, int(total * every)), 0
    for k, x in enumerate(it, 1):
        if k >= nxt or k == total:
            log(f"  {what}: {k}/{total}")
            nxt = k + step
        yield x


# ---------- LOL ----------

def lol_pairs(inp):
    v1 = find_dir(inp, "LOLv1")
    v2 = os.path.join(find_dir(inp, "LOLv2"), "Real_captured")
    out = []
    for src, base in (("v1", v1), ("v2", v2)):
        for split in ("Train", "Test"):
            lo, hi = os.path.join(base, split, "Low"), os.path.join(base, split, "Normal")
            for f in pngs(hi):
                if not os.path.exists(os.path.join(lo, f)):
                    continue
                out.append(dict(id=f"{src}_{split.lower()}_{os.path.splitext(f)[0]}", src=src,
                                official=split.lower(), low=os.path.join(lo, f), high=os.path.join(hi, f)))
    return out


def thumb(x):
    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    t = cv2.resize(g, (48, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((t - t.mean()) / (t.std() + 1e-6)).ravel()


def raw_thumb(x):
    return cv2.resize(x, (48, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def split_lol(pairs, highs, lows, rng):
    """LOL shoots one scene at several dark exposures and pairs each with the same bright photo, so a shared
    bright photo is NOT a duplicate pair. A pair is a copy only if its dark AND bright images both match.
    Leakage is handled at scene level: train pairs whose bright photo is close to any test image are dropped."""
    n = len(pairs)
    T = np.stack([thumb(h) for h in highs])
    L = np.stack([raw_thumb(x) for x in lows])

    cache = {}

    def is_copy(i, k):
        """Same bright photo AND the same dark photo pixel for pixel (a separate shot has its own noise).
        Dark frames are so dark that thumbnails can't tell exposures apart, so compare at full size."""
        if D[i, k] >= LOL_COPY or float(np.abs(L[i] - L[k]).mean()) > 3.0:
            return False
        key = (min(i, k), max(i, k))
        if key not in cache:
            a, b = lows[i], lows[k]
            cache[key] = a.shape == b.shape and float(np.abs(a.astype(np.int16) - b).mean()) < LOL_LOW_COPY
        return cache[key]
    D = np.empty((n, n), np.float32)
    for i in range(n):  # row by row: a full n x n x 1536 broadcast would need ~10 GB
        D[i] = np.abs(T - T[i]).mean(1)
    np.fill_diagonal(D, np.inf)

    test_all = [i for i in range(n) if pairs[i]["official"] == "test"]
    train_all = [i for i in range(n) if pairs[i]["official"] == "train"]
    dropped = []

    # test: keep every official test image except exact copies of one already kept
    test = []
    for i in test_all:
        j = next((k for k in test if is_copy(i, k)), None)
        if j is None:
            test.append(i)
        else:
            dropped.append((i, f"copy of test image {pairs[j]['id']}", float(D[i, j])))

    # train: drop anything close to any test image (kept or copy), then copies inside train
    near_test = D[np.ix_(train_all, test_all)].min(1) if train_all else np.array([])
    cand = []
    for i, d in zip(train_all, near_test):
        if d < LOL_SCENE:
            k = test_all[int(np.argmin(D[i, test_all]))]
            dropped.append((i, f"same scene as test image {pairs[k]['id']}", float(d)))
        else:
            cand.append(i)
    keep = []
    for i in cand:
        j = next((k for k in keep if is_copy(i, k)), None)
        if j is None:
            keep.append(i)
        else:
            dropped.append((i, f"copy of train image {pairs[j]['id']}", float(D[i, j])))

    # val: random seeds plus their close neighbours (one hop), so val scenes are mostly absent from train
    seeds = [keep[k] for k in rng.permutation(len(keep))[: min(LOL_VAL_SEEDS, len(keep) // 5)]]
    vs = set(seeds)
    for i in keep:
        if i not in vs and min(D[i, s] for s in seeds) < LOL_SCENE:
            vs.add(i)
    val = [i for i in keep if i in vs]
    train = [i for i in keep if i not in vs]

    v1 = [i for i in range(n) if pairs[i]["src"] == "v1"]
    v2 = [i for i in range(n) if pairs[i]["src"] == "v2"]
    v1_in_v2 = sum(any(is_copy(i, k) for k in v2 if D[i, k] < LOL_COPY) for i in v1)
    shared = int(sum((D[i] < LOL_COPY).any() for i in range(n)))
    hist = lambda a: np.histogram(np.minimum(a, 1.5), bins=15, range=(0, 1.5))[0].tolist()
    test_test = [float(D[i, [k for k in test_all if k != i]].min()) for i in test_all] if len(test_all) > 1 else []
    report = dict(
        copy_threshold=LOL_COPY, low_copy_threshold=LOL_LOW_COPY, scene_threshold=LOL_SCENE,
        pairs_sharing_a_bright_photo=shared,
        found=dict(v1=len(v1), v2=len(v2), official_test=len(test_all), official_train=len(train_all)),
        kept=dict(train=len(train), val=len(val), test=len(test)),
        v1_pairs_copied_in_v2=v1_in_v2,
        dropped_counts={w: sum(1 for _, x, _ in dropped if x.startswith(w))
                        for w in ("copy of test", "same scene as test", "copy of train")},
        dropped=[dict(id=pairs[i]["id"], why=w, dist=round(d, 3)) for i, w, d in dropped],
        hist_bins="0 to 1.5 in steps of 0.1",
        hist_train_to_nearest_test=hist(near_test) if len(near_test) else [],
        hist_test_to_nearest_other_test=hist(np.array(test_test)) if test_test else [],
    )

    # picture: train/test pairs just under and just over the scene threshold, to eyeball it
    order = np.argsort(near_test) if len(near_test) else []
    under = [r for r in order if near_test[r] < LOL_SCENE][-5:]
    over = [r for r in order if near_test[r] >= LOL_SCENE][:5]
    rows = []
    for r in list(under) + list(over):
        i = train_all[r]
        k = test_all[int(np.argmin(D[i, test_all]))]
        a = cv2.resize(highs[i], (240, 160), interpolation=cv2.INTER_AREA)
        b = cv2.resize(highs[k], (240, 160), interpolation=cv2.INTER_AREA)
        row = np.concatenate([a, b], 1).copy()
        tag = f"{near_test[r]:.2f} {'DROPPED' if near_test[r] < LOL_SCENE else 'kept'}"
        cv2.putText(row, tag, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        rows.append(row)
    check = np.concatenate(rows, 0) if rows else None
    return train, val, test, report, check


# ---------- SIDD ----------

def sidd_instances(inp):
    d = os.path.join(find_dir(inp, "SIDD_Small_sRGB_Only"), "Data")
    out = []
    for name in sorted(os.listdir(d)):
        parts = name.split("_")
        files = os.listdir(os.path.join(d, name))
        gt = [f for f in files if f.upper().startswith("GT_SRGB")]
        no = [f for f in files if f.upper().startswith("NOISY_SRGB")]
        if not gt or not no:
            continue
        out.append(dict(id=name, scene=parts[1], phone=parts[2], iso=int(parts[3]),
                        gt=os.path.join(d, name, gt[0]), noisy=os.path.join(d, name, no[0])))
    return out


def job_sidd(args):
    """Read one SIDD pair and write its crops: train -> rows of the .npy files, val/test -> PNGs."""
    inst, split, pos, start, out, stage = args
    cv2.setNumThreads(1)
    g, z = imread(inst["gt"]), imread(inst["noisy"])
    if split == "train":
        an = np.load(os.path.join(out, "train_sidd_noisy.npy"), mmap_mode="r+")
        ag = np.load(os.path.join(out, "train_sidd_gt.npy"), mmap_mode="r+")
        for k, (y, x) in enumerate(pos):
            an[start + k] = z[y:y + PATCH, x:x + PATCH]
            ag[start + k] = g[y:y + PATCH, x:x + PATCH]
        an.flush(), ag.flush()
    else:
        for k, (y, x) in enumerate(pos):
            name = f"{inst['id']}_{k:02d}.png"
            imwrite(os.path.join(stage, split, "noisy/input", name), z[y:y + PATCH, x:x + PATCH])
            imwrite(os.path.join(stage, split, "noisy/target", name), g[y:y + PATCH, x:x + PATCH])
    return inst["id"]


# ---------- DIV2K ----------

def div2k_train_positions(h, w, rng):
    full = grid_positions(h, w, PATCH, DIV2K_FULL_PATCHES, rng)
    half = grid_positions(h // 2, w // 2, PATCH, 2, rng)
    return [(1, y, x) for y, x in full] + [(2, y, x) for y, x in half]


def job_div2k_train(args):
    path, pos, start, out = args
    cv2.setNumThreads(1)
    x = imread(path)
    h, w = x.shape[:2]
    half = cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    a = np.load(os.path.join(out, "train_clean.npy"), mmap_mode="r+")
    for k, (s, y, xx) in enumerate(pos):
        im = x if s == 1 else half
        a[start + k] = im[y:y + PATCH, xx:xx + PATCH]
    a.flush()
    return path


def job_div2k_fixed(args):
    """Fixed degraded copies for val/test: old and synthetic-noise inputs with per-image seeds."""
    path, stage, split, seed, center_crop = args
    cv2.setNumThreads(1)
    stem = os.path.splitext(os.path.basename(path))[0]
    x = resize_long(imread(path), TEST_LONG_SIDE)
    if center_crop:
        h, w = x.shape[:2]
        y0, x0 = (h - PATCH) // 2, (w - PATCH) // 2
        x = x[y0:y0 + PATCH, x0:x0 + PATCH]
    x = crop16(x)
    clean = x.astype(np.float32) / 255
    rec = {}
    for task, fn, off in (("old", degrade.old, 0), ("noisy_syn", degrade.noisy, 1)):
        y, params = fn(clean, np.random.default_rng([seed, off]))
        imwrite(os.path.join(stage, split, task, "input", stem + ".png"), y)
        imwrite(os.path.join(stage, split, task, "target", stem + ".png"), x)
        rec[task] = params
    return stem, rec


# ---------- output helpers ----------

def file_md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def preview(stage, dst):
    """One test example per task (input over target), to eyeball the data."""
    tiles = []
    for task in ("dark", "noisy", "old", "noisy_syn"):
        d = os.path.join(stage, "test", task)
        f = pngs(os.path.join(d, "input"))[0]
        pair = [cv2.resize(imread(os.path.join(d, k, f)), (320, 240), interpolation=cv2.INTER_AREA)
                for k in ("input", "target")]
        tiles.append(np.concatenate(pair, 0))
    cv2.imwrite(dst, cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/bareq-data")
    ap.add_argument("--stage", default="/tmp/bareq-stage", help="scratch folder for the val/test PNG trees")
    ap.add_argument("--workers", type=int, default=0, help="0 = twice the CPUs, at most 8")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only a few images per source")
    a = ap.parse_args()
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
    workers = a.workers or min(8, 2 * cpus)  # more processes than CPUs hides the input mount's latency
    rng = np.random.default_rng(SEED)
    out, stage = a.out, a.stage
    for d in (out, stage):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    man = dict(seed=SEED, created=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
               numpy=np.__version__, opencv=cv2.__version__, limit=a.limit, cpus=cpus, workers=workers,
               git_commit=os.environ.get("BAREQ_COMMIT", "unknown"))
    log(f"cpus {cpus}, workers {workers}")

    # ----- LOL -----
    pairs = lol_pairs(a.input)
    if a.limit:
        pairs = [p for p in pairs if p["official"] == "test"][: a.limit] + \
                [p for p in pairs if p["official"] == "train"][: a.limit * 3]
    log(f"LOL: {len(pairs)} pairs found, reading")
    highs = [crop16(x) for x in read_many([p["high"] for p in pairs])]
    lows = [crop16(x) for x in read_many([p["low"] for p in pairs])]
    tr, va, te, rep, check = split_lol(pairs, highs, lows, rng)
    if check is not None:
        cv2.imwrite(os.path.join(out, "dedupe_check.jpg"), cv2.cvtColor(check, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    log(f"LOL split: {rep['kept']}, dropped {rep['dropped_counts']}, v1 pairs copied in v2: {rep['v1_pairs_copied_in_v2']}, pairs sharing a bright photo: {rep['pairs_sharing_a_bright_photo']}")
    shape = highs[tr[0]].shape
    new_npy(os.path.join(out, "train_dark_low.npy"), len(tr), shape)
    new_npy(os.path.join(out, "train_dark_high.npy"), len(tr), shape)
    al = np.load(os.path.join(out, "train_dark_low.npy"), mmap_mode="r+")
    ah = np.load(os.path.join(out, "train_dark_high.npy"), mmap_mode="r+")
    resized = 0
    for r, i in enumerate(tr):
        lo, hi = lows[i], highs[i]
        if hi.shape != shape:
            lo = cv2.resize(lo, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            hi = cv2.resize(hi, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            resized += 1
        al[r], ah[r] = lo, hi
    al.flush(), ah.flush()
    del al, ah
    for split, rows in (("val", va), ("test", te)):
        for i in rows:
            imwrite(os.path.join(stage, split, "dark/input", pairs[i]["id"] + ".png"), lows[i])
            imwrite(os.path.join(stage, split, "dark/target", pairs[i]["id"] + ".png"), highs[i])
    man["lol"] = dict(train=[pairs[i]["id"] for i in tr], val=[pairs[i]["id"] for i in va],
                      test=[pairs[i]["id"] for i in te], train_shape=list(shape), train_resized=resized, dedupe=rep)
    del highs, lows
    log("LOL written")

    # ----- SIDD -----
    inst = sidd_instances(a.input)
    if a.limit:
        inst = inst[: a.limit * 3]
    sizes = read_many([i["gt"] for i in inst], fn=img_size)
    split_of = lambda s: "test" if s in SIDD_TEST_SCENES else "val" if s in SIDD_VAL_SCENES else "train"
    jobs, start = [], 0
    for k, (i, (h, w)) in enumerate(zip(inst, sizes)):
        sp = split_of(i["scene"])
        n = {"train": SIDD_TRAIN_PATCHES, "val": SIDD_VAL_CROPS, "test": SIDD_TEST_CROPS}[sp]
        pos = grid_positions(h, w, PATCH, n, np.random.default_rng([SEED, 1000 + k]))
        jobs.append((i, sp, pos, start if sp == "train" else -1, out, stage))
        if sp == "train":
            start += len(pos)
    new_npy(os.path.join(out, "train_sidd_noisy.npy"), start, (PATCH, PATCH, 3))
    new_npy(os.path.join(out, "train_sidd_gt.npy"), start, (PATCH, PATCH, 3))
    log(f"SIDD: {len(inst)} images, {start} train patches")
    with Pool(workers) as pool:
        for _ in progress(pool.imap_unordered(job_sidd, jobs), len(jobs), "SIDD"):
            pass
    man["sidd"] = {sp: dict(scenes=sorted({j[0]["scene"] for j in jobs if j[1] == sp}),
                            instances=[j[0]["id"] for j in jobs if j[1] == sp],
                            crops=sum(len(j[2]) for j in jobs if j[1] == sp))
                   for sp in ("train", "val", "test")}
    man["sidd"]["train_rows"] = [dict(id=j[0]["id"], start=j[3], n=len(j[2])) for j in jobs if j[1] == "train"]
    man["sidd"]["test_crops"] = [dict(id=j[0]["id"], pos=j[2]) for j in jobs if j[1] == "test"]

    # ----- DIV2K -----
    dtrain, dvalid = find_dir(a.input, "DIV2K_train_HR"), find_dir(a.input, "DIV2K_valid_HR")
    if not pngs(dtrain):  # some uploads nest the folder twice
        dtrain = os.path.join(dtrain, "DIV2K_train_HR")
    if not pngs(dvalid):
        dvalid = os.path.join(dvalid, "DIV2K_valid_HR")
    trn = [os.path.join(dtrain, f) for f in pngs(dtrain)]
    tst = [os.path.join(dvalid, f) for f in pngs(dvalid)]
    if a.limit:
        trn, tst = trn[: a.limit * 2], tst[: a.limit]
    perm = rng.permutation(len(trn))
    nval = min(DIV2K_VAL, max(1, len(trn) // 10))
    val_files = sorted(trn[i] for i in perm[:nval])
    trn_files = sorted(trn[i] for i in perm[nval:])
    sizes = read_many(trn_files, fn=img_size)
    jobs, start = [], 0
    for k, (p, (h, w)) in enumerate(zip(trn_files, sizes)):
        pos = div2k_train_positions(h, w, np.random.default_rng([SEED, 2000 + k]))
        jobs.append((p, pos, start, out))
        start += len(pos)
    new_npy(os.path.join(out, "train_clean.npy"), start, (PATCH, PATCH, 3))
    log(f"DIV2K: {len(trn_files)} train images -> {start} patches, val {len(val_files)}, test {len(tst)}")
    fixed = [(p, stage, "val", SEED + 3000 + k, True) for k, p in enumerate(val_files)] + \
            [(p, stage, "test", SEED + 4000 + k, False) for k, p in enumerate(tst)]
    with Pool(workers) as pool:
        for _ in progress(pool.imap_unordered(job_div2k_train, jobs), len(jobs), "DIV2K train"):
            pass
        params = dict(progress(pool.imap_unordered(job_div2k_fixed, fixed), len(fixed), "DIV2K val/test"))
    man["div2k"] = dict(
        train=[os.path.basename(p) for p in trn_files], val=[os.path.basename(p) for p in val_files],
        test=[os.path.basename(p) for p in tst], train_patches=start,
        train_rows=[dict(file=os.path.basename(j[0]), start=j[2], n=len(j[1])) for j in jobs],
        params={k: params[k] for k in sorted(params)},
        note=f"val: {PATCH}px centre crops; test: full image resized to long side {TEST_LONG_SIDE}")

    # ----- checks -----
    leaks = []
    for name, d in (("lol", man["lol"]), ("div2k", man["div2k"])):
        for s1, s2 in (("train", "val"), ("train", "test"), ("val", "test")):
            both = set(d[s1]) & set(d[s2])
            if both:
                leaks.append(f"{name} {s1}/{s2}: {sorted(both)[:5]}")
    for s1, s2 in (("train", "val"), ("train", "test"), ("val", "test")):
        both = set(man["sidd"][s1]["scenes"]) & set(man["sidd"][s2]["scenes"])
        if both:
            leaks.append(f"sidd scenes {s1}/{s2}: {sorted(both)}")
    man["leak_check"] = leaks or "ok"
    log("leak check:", man["leak_check"])
    if leaks:
        sys.exit("LEAK FOUND")

    # ----- pack val/test -----
    counts = {}
    for root, _, files in os.walk(stage):
        n = sum(f.endswith(".png") for f in files)
        if n:
            counts[os.path.relpath(root, stage)] = n
    for f in sorted(os.listdir(out)):
        if f.endswith(".npy"):
            counts[f] = int(np.load(os.path.join(out, f), mmap_mode="r").shape[0])
    man["counts"] = dict(sorted(counts.items()))
    man["test_md5"] = {os.path.relpath(os.path.join(r, f), stage): file_md5(os.path.join(r, f))
                       for r, _, fs in os.walk(os.path.join(stage, "test")) for f in sorted(fs)}
    for split in ("val", "test"):
        with tarfile.open(os.path.join(out, f"{split}.tar"), "w") as t:
            t.add(os.path.join(stage, split), arcname=split)
    preview(stage, os.path.join(out, "preview.jpg"))
    man["seconds"] = round(time.time() - T0)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    for k, v in man["counts"].items():
        print(f"  {v:6}  {k}")
    for f in sorted(os.listdir(out)):
        print(f"  {os.path.getsize(os.path.join(out, f)) / 1e6:9.1f} MB  {f}")
    log("done")


if __name__ == "__main__":
    main()
