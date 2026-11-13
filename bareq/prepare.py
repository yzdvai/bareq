"""Build the Bareq dataset: split first, then crop and degrade.

  python -m bareq.prepare --input /kaggle/input --out /kaggle/working/bareq-data

Layout written to --out (all PNG, 8-bit RGB):
  train/dark/{low,high}/        LOL v1 + v2-Real pairs, full size (cropped randomly at train time)
  train/sidd/{noisy,gt}/        SIDD Small sRGB, 512px patches, train scenes only
  train/clean/                  DIV2K 512px patches (noisy + old inputs are made on the fly while training)
  val/{dark,noisy,old,noisy_syn}/{input,target}/   fixed, small
  test/{dark,noisy,old,noisy_syn}/{input,target}/  fixed; generated once with fixed seeds
  manifest.json                 every split, the LOL duplicate report, seeds, counts, degradation params

Leakage rules:
  - LOL: v1 and v2-Real are pooled; near-duplicate scenes are grouped (union-find on thumbnails of the
    normal-light image), and any group touching a test image is removed from train and val.
  - SIDD: split by scene number, so a scene is never in two splits.
  - DIV2K: one split shared by synthetic noise and "old", official valid set (0801-0900) is the test set.
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
from multiprocessing import Pool

import cv2
import numpy as np

from . import degrade

SEED = 20261113
SIDD_TEST_SCENES = ["003", "005"]   # all five phones, ISO up to 10000
SIDD_VAL_SCENES = ["004"]
SIDD_TRAIN_PATCHES = 24             # per image, 512px, non-overlapping
SIDD_TEST_CROPS = 8                 # per image, 512px
SIDD_VAL_CROPS = 4
DIV2K_VAL = 40                      # taken from the 800 train images
LOL_VAL_GROUPS = 30
LOL_DUP_THRESHOLD = 0.35            # mean abs diff of normalised 48x32 thumbnails; same scene << 0.35 << different scene
PATCH = 512
TEST_LONG_SIDE = 1024               # DIV2K test/val images are resized to this before degrading


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


def thumb(p):
    x = cv2.cvtColor(imread(p), cv2.COLOR_RGB2GRAY)
    t = cv2.resize(x, (48, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((t - t.mean()) / (t.std() + 1e-6)).ravel()


def split_lol(pairs, rng):
    T = np.stack([thumb(p["high"]) for p in pairs])
    n = len(pairs)
    D = np.empty((n, n), np.float32)
    for i in range(n):  # row by row: the full n x n x 1536 broadcast would need ~10 GB
        D[i] = np.abs(T - T[i]).mean(1)
    np.fill_diagonal(D, np.inf)

    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(D < LOL_DUP_THRESHOLD)):
        parent[root(i)] = root(j)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)

    is_test = [p["official"] == "test" for p in pairs]
    test, dropped, trainable = [], [], []
    for members in groups.values():
        if any(is_test[i] for i in members):
            tests = [i for i in members if is_test[i]]
            test.append(tests[0])                       # one test image per scene group
            dropped += [(i, "duplicate of another test image") for i in tests[1:]]
            dropped += [(i, "same scene as a test image") for i in members if not is_test[i]]
        else:
            trainable.append(sorted(members))
    order = rng.permutation(len(trainable))
    val = [i for k in order[:LOL_VAL_GROUPS] for i in trainable[k]]
    train = [i for k in order[LOL_VAL_GROUPS:] for i in trainable[k]]

    # report: nearest distance from each test image to anything not in the test split
    others = np.array([not t for t in is_test])
    near = sorted(float(D[i][others].min()) for i in range(n) if is_test[i])
    cross = [(pairs[i]["id"], pairs[j]["id"], round(float(D[i, j]), 3))
             for i, j in zip(*np.where(D < LOL_DUP_THRESHOLD)) if i < j and pairs[i]["src"] != pairs[j]["src"]]
    report = dict(
        threshold=LOL_DUP_THRESHOLD,
        groups=len(groups),
        multi_member_groups=sum(len(g) > 1 for g in groups.values()),
        v1_v2_near_duplicate_pairs=len(cross),
        v1_v2_examples=cross[:20],
        dropped=[dict(id=pairs[i]["id"], why=w) for i, w in dropped],
        test_nearest_nontest_distance=dict(
            min=round(near[0], 3), p10=round(near[len(near) // 10], 3), median=round(near[len(near) // 2], 3)),
        all_nearest_distance_hist=np.histogram(np.minimum(D.min(1), 2.0), bins=20, range=(0, 2))[0].tolist(),
    )
    return [pairs[i] for i in train], [pairs[i] for i in val], [pairs[i] for i in test], report


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


def grid_positions(h, w, size, n, rng):
    """Up to n non-overlapping size x size crops on a grid, chosen at random."""
    ys, xs = np.arange(0, h - size + 1, size), np.arange(0, w - size + 1, size)
    pos = [(y, x) for y in ys for x in xs]
    idx = rng.permutation(len(pos))[:n]
    return [pos[i] for i in sorted(idx)]


def job_sidd(args):
    inst, out, split, n, seed = args
    cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    g, z = imread(inst["gt"]), imread(inst["noisy"])
    rows = []
    for k, (y, x) in enumerate(grid_positions(*g.shape[:2], PATCH, n, rng)):
        name = f"{inst['id']}_{k:02d}.png"
        if split == "train":
            imwrite(os.path.join(out, "train/sidd/noisy", name), z[y:y + PATCH, x:x + PATCH])
            imwrite(os.path.join(out, "train/sidd/gt", name), g[y:y + PATCH, x:x + PATCH])
        else:
            imwrite(os.path.join(out, split, "noisy/input", name), z[y:y + PATCH, x:x + PATCH])
            imwrite(os.path.join(out, split, "noisy/target", name), g[y:y + PATCH, x:x + PATCH])
        rows.append(dict(file=name, src=inst["id"], y=int(y), x=int(x)))
    return rows


# ---------- DIV2K ----------

def job_div2k_train(args):
    path, out, seed = args
    cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    x = imread(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    n = 0
    for scale in (1.0, 0.5):  # full size and half size, so grain and noise are seen at both scales
        im = x if scale == 1.0 else cv2.resize(x, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        for k, (y, xx) in enumerate(grid_positions(*im.shape[:2], PATCH, 6, rng)):
            imwrite(os.path.join(out, "train/clean", f"{stem}_s{int(scale * 10)}_{k}.png"),
                    im[y:y + PATCH, xx:xx + PATCH])
            n += 1
    return n


def job_div2k_fixed(args):
    """Fixed degraded copies for val/test: old and synthetic-noise inputs with per-image seeds."""
    path, out, split, seed, center_crop = args
    cv2.setNumThreads(1)
    stem = os.path.splitext(os.path.basename(path))[0]
    x = imread(path)
    x = resize_long(x, TEST_LONG_SIDE)
    if center_crop:
        h, w = x.shape[:2]
        y0, x0 = (h - PATCH) // 2, (w - PATCH) // 2
        x = x[y0:y0 + PATCH, x0:x0 + PATCH]
    x = crop16(x)
    clean = x.astype(np.float32) / 255
    rec = {}
    for task, fn, off in (("old", degrade.old, 0), ("noisy_syn", degrade.noisy, 1)):
        rng = np.random.default_rng([seed, off])
        y, params = fn(clean, rng)
        imwrite(os.path.join(out, split, task, "input", stem + ".png"), y)
        imwrite(os.path.join(out, split, task, "target", stem + ".png"), x)
        rec[task] = params
    return stem, rec


# ---------- packing ----------

def pack_npy(src_dir, dst, shape, log):
    """Stack same-size PNGs into one uint8 .npy (N x H x W x 3), written via memmap to keep RAM low."""
    files = pngs(src_dir)
    arr = np.lib.format.open_memmap(dst, mode="w+", dtype=np.uint8, shape=(len(files),) + shape)
    resized = 0
    for i, f in enumerate(files):
        x = imread(os.path.join(src_dir, f))
        if x.shape != shape:
            x = cv2.resize(x, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            resized += 1
        arr[i] = x
    arr.flush()
    del arr
    log(f"  {os.path.basename(dst)}: {len(files)} x {shape}" + (f" ({resized} resized)" if resized else ""))
    return dict(file=os.path.basename(dst), n=len(files), shape=list(shape), names=files, resized=resized)


def pack(stage, dst, log):
    """Training sets -> .npy (fast random crops, no decoding); val/test -> one .tar each (PNG trees)."""
    os.makedirs(dst, exist_ok=True)
    out = {}
    lol_shape = imread(os.path.join(stage, "train/dark/high", pngs(os.path.join(stage, "train/dark/high"))[0])).shape
    for key, sub, shape in (("dark_low", "train/dark/low", lol_shape), ("dark_high", "train/dark/high", lol_shape),
                            ("sidd_noisy", "train/sidd/noisy", (PATCH, PATCH, 3)),
                            ("sidd_gt", "train/sidd/gt", (PATCH, PATCH, 3)),
                            ("clean", "train/clean", (PATCH, PATCH, 3))):
        out[key] = pack_npy(os.path.join(stage, sub), os.path.join(dst, f"train_{key}.npy"), shape, log)
        shutil.rmtree(os.path.join(stage, sub))  # free disk as we go
    assert out["dark_low"]["names"] == out["dark_high"]["names"]
    assert out["sidd_noisy"]["names"] == out["sidd_gt"]["names"]
    for split in ("val", "test"):
        with tarfile.open(os.path.join(dst, f"{split}.tar"), "w") as t:
            t.add(os.path.join(stage, split), arcname=split)
        log(f"  {split}.tar")
    return out


def preview(stage, dst):
    """One test example per task (input over target), to eyeball the data."""
    tiles = []
    for task in ("dark", "noisy", "old", "noisy_syn"):
        d = os.path.join(stage, "test", task)
        f = pngs(os.path.join(d, "input"))[0]
        pair = [cv2.resize(imread(os.path.join(d, k, f)), (320, 240), interpolation=cv2.INTER_AREA)
                for k in ("input", "target")]
        tiles.append(np.concatenate(pair, 0))
    g = np.concatenate(tiles, 1)
    cv2.imwrite(dst, cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------- main ----------

def file_md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/bareq-data")
    ap.add_argument("--stage", default="/tmp/bareq-stage", help="scratch folder for the PNG tree before packing")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many images per source")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out = a.stage  # PNG tree is built here, then packed into a.out
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    man = dict(seed=SEED, created=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
               numpy=np.__version__, opencv=cv2.__version__, limit=a.limit,
               git_commit=os.environ.get("BAREQ_COMMIT", "unknown"))

    def log(*s):
        print(f"[{time.time() - t0:6.0f}s]", *s, flush=True)

    # LOL
    pairs = lol_pairs(a.input)
    if a.limit:
        pairs = [p for p in pairs if p["official"] == "test"][: a.limit] + \
                [p for p in pairs if p["official"] == "train"][: a.limit * 3]
    log(f"LOL: {len(pairs)} pairs found")
    tr, va, te, rep = split_lol(pairs, rng)
    log(f"LOL split: train {len(tr)}, val {len(va)}, test {len(te)}, dropped {len(rep['dropped'])}, "
        f"v1/v2 near-duplicate pairs {rep['v1_v2_near_duplicate_pairs']}")
    for split, rows in (("train", tr), ("val", va), ("test", te)):
        for p in rows:
            lo, hi = crop16(imread(p["low"])), crop16(imread(p["high"]))
            if split == "train":
                imwrite(os.path.join(out, "train/dark/low", p["id"] + ".png"), lo)
                imwrite(os.path.join(out, "train/dark/high", p["id"] + ".png"), hi)
            else:
                imwrite(os.path.join(out, split, "dark/input", p["id"] + ".png"), lo)
                imwrite(os.path.join(out, split, "dark/target", p["id"] + ".png"), hi)
    man["lol"] = dict(train=[p["id"] for p in tr], val=[p["id"] for p in va], test=[p["id"] for p in te],
                      dedupe=rep)

    # SIDD
    inst = sidd_instances(a.input)
    if a.limit:
        inst = inst[: a.limit * 3]
    split_of = lambda s: "test" if s in SIDD_TEST_SCENES else "val" if s in SIDD_VAL_SCENES else "train"
    jobs = []
    for k, i in enumerate(inst):
        sp = split_of(i["scene"])
        n = {"train": SIDD_TRAIN_PATCHES, "val": SIDD_VAL_CROPS, "test": SIDD_TEST_CROPS}[sp]
        jobs.append((i, out, sp, n, SEED + 1000 + k))
    with Pool(a.workers) as pool:
        res = pool.map(job_sidd, jobs, chunksize=1)
    man["sidd"] = {sp: dict(scenes=sorted({j[0]["scene"] for j in jobs if j[2] == sp}),
                            instances=[j[0]["id"] for j in jobs if j[2] == sp],
                            files=sum(len(r) for r, j in zip(res, jobs) if j[2] == sp))
                   for sp in ("train", "val", "test")}
    man["sidd"]["test_crops"] = [row for r, j in zip(res, jobs) if j[2] == "test" for row in r]
    log("SIDD:", {sp: (len(v["instances"]), v["files"]) for sp, v in man["sidd"].items() if sp != "test_crops"})

    # DIV2K
    dtrain = find_dir(a.input, "DIV2K_train_HR")
    dvalid = find_dir(a.input, "DIV2K_valid_HR")
    # some uploads nest the folder twice (DIV2K_train_HR/DIV2K_train_HR)
    if not pngs(dtrain):
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
    with Pool(a.workers) as pool:
        counts = pool.map(job_div2k_train, [(p, out, SEED + 2000 + k) for k, p in enumerate(trn_files)], chunksize=2)
        fixed_val = pool.map(job_div2k_fixed, [(p, out, "val", SEED + 3000 + k, True) for k, p in enumerate(val_files)])
        fixed_test = pool.map(job_div2k_fixed, [(p, out, "test", SEED + 4000 + k, False) for k, p in enumerate(tst)])
    man["div2k"] = dict(
        train=[os.path.basename(p) for p in trn_files], val=[os.path.basename(p) for p in val_files],
        test=[os.path.basename(p) for p in tst], train_patches=int(sum(counts)),
        val_params=dict(fixed_val), test_params=dict(fixed_test),
        note=f"val: {PATCH}px centre crops; test: full image resized to long side {TEST_LONG_SIDE}")
    log(f"DIV2K: train {len(trn_files)} images -> {sum(counts)} patches, val {len(val_files)}, test {len(tst)}")

    # checks: no file name appears in two splits, and the test images' content hashes
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

    counts = {}
    for root, _, files in os.walk(out):
        n = sum(f.endswith(".png") for f in files)
        if n:
            counts[os.path.relpath(root, out)] = n
    man["counts"] = dict(sorted(counts.items()))
    man["test_md5"] = {os.path.relpath(os.path.join(r, f), out): file_md5(os.path.join(r, f))
                       for r, _, fs in os.walk(os.path.join(out, "test")) for f in sorted(fs)}
    log("leak check:", man["leak_check"])
    for k, v in man["counts"].items():
        print(f"  {v:6}  {k}")
    if leaks:
        sys.exit("LEAK FOUND, see manifest.json")

    log("packing into", a.out)
    man["packed"] = pack(out, a.out, log)
    preview(out, os.path.join(a.out, "preview.jpg"))
    man["seconds"] = round(time.time() - t0)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    for f in sorted(os.listdir(a.out)):
        print(f"  {os.path.getsize(os.path.join(a.out, f)) / 1e6:9.1f} MB  {f}")
    log("done")


if __name__ == "__main__":
    main()
