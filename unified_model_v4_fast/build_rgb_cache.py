#!/usr/bin/env python3
"""Pre-decode RGB PNGs to uint8 .npy. Lossless: PNG is lossless and the
original loader does np.asarray(Image.open(p).convert("RGB")) before scaling,
so the cached array is bit-identical to what training would have decoded."""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image


def one(job):
    src, dst = job
    if os.path.exists(dst):
        return 0
    a = np.asarray(Image.open(src).convert("RGB"))      # (h,w,3) uint8
    tmp = dst + ".tmp"
    with open(tmp, "wb") as fh:          # file handle: np.save must not append .npy
        np.save(fh, a)
    os.replace(tmp, dst)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    src = sorted(glob.glob(os.path.join(a.png_dir, "*.png")))
    if not src:
        raise SystemExit(f"no PNGs in {a.png_dir}")
    # keep sorted order identical: same basename, .npy extension
    jobs = [(p, os.path.join(a.out_dir, os.path.splitext(os.path.basename(p))[0] + ".npy"))
            for p in src]
    with ProcessPoolExecutor(a.workers) as ex:
        made = sum(ex.map(one, jobs, chunksize=16))
    out = sorted(glob.glob(os.path.join(a.out_dir, "*.npy")))
    print(f"rgb cache: {len(out)} npy ({made} newly decoded) from {len(src)} png in {a.out_dir}")
    assert len(out) == len(src), f"cache incomplete: {len(out)} != {len(src)}"
    # verify sorted order corresponds 1:1 so index i means the same frame
    for p, q in zip(src, out):
        assert os.path.splitext(os.path.basename(p))[0] == os.path.splitext(os.path.basename(q))[0]
    print("sorted-order correspondence verified")


if __name__ == "__main__":
    main()
