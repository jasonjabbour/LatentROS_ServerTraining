#!/usr/bin/env python3
"""Portable multimodal data loading with the original v2 sampling and derivation.

Depth and RGB are read directly. Occupancy comes from --derived-dir; costmap
and heatmap are derived from that occupancy on demand. Training never rebuilds
the cache. Use the explicit --prep command only when it does not exist yet.
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEPTH_DIR = RGB_DIR = OCC_CACHE = None
GRID_MODS = ("depth", "occupancy", "costmap", "heatmap")
MOD_NCH = {"depth": 1, "occupancy": 1, "costmap": 1, "heatmap": 1, "rgb": 3}


# ---------------------------------------------------------------- derivation (from depth)
def bev_occupancy(depth, gx=200, gz=200, xmax=20.0, zmax=40.0, hfov=90.0):
    Hd, Wd = depth.shape
    fx = Wd / (2 * np.tan(np.radians(hfov) / 2)); cx, cy = Wd / 2, Hd / 2
    us, vs = np.meshgrid(np.arange(Wd), np.arange(Hd))
    Z = depth; m = (Z > 1.5) & (Z < zmax)
    X = (us - cx) * Z / fx; Y = (vs - cy) * Z / fx
    m &= (Y > -1.0) & (Y < 2.5)
    ix = ((X[m] + xmax) / (2 * xmax) * gx).astype(int); iz = (Z[m] / zmax * gz).astype(int)
    ok = (ix >= 0) & (ix < gx) & (iz >= 0) & (iz < gz)
    grid = np.zeros((gz, gx), np.float32); np.add.at(grid, (iz[ok], ix[ok]), 1.0)
    return np.ascontiguousarray((grid > 2).astype(np.float32)[::-1])


def inflate(occ, rings=22):
    cost = occ.astype(np.float32).copy(); cur = occ.astype(bool).copy()
    for k in range(1, rings + 1):
        nb = cur.copy()
        nb[1:, :] |= cur[:-1, :]; nb[:-1, :] |= cur[1:, :]
        nb[:, 1:] |= cur[:, :-1]; nb[:, :-1] |= cur[:, 1:]
        ring = nb & ~cur; cost[ring] = np.maximum(cost[ring], 1.0 - k / (rings + 1)); cur = nb
    return cost


def blur(v, r=8, passes=3):
    k = np.ones(2 * r + 1, np.float32) / (2 * r + 1); out = v.astype(np.float32).copy()
    for _ in range(passes):
        out = np.apply_along_axis(lambda a: np.convolve(a, k, 'same'), 0, out)
        out = np.apply_along_axis(lambda a: np.convolve(a, k, 'same'), 1, out)
    return out


# ---------------------------------------------------------------- file lists + cache
def configure(depth_dir, rgb_dir, derived_dir):
    """Set explicit dataset locations before preparing or loading data."""
    global DEPTH_DIR, RGB_DIR, OCC_CACHE
    DEPTH_DIR, RGB_DIR, OCC_CACHE = [os.path.abspath(os.path.expanduser(os.fspath(p)))
                                   for p in (depth_dir, rgb_dir, derived_dir)]


def file_lists():
    if DEPTH_DIR is None:
        raise RuntimeError("Call configure(depth_dir, rgb_dir, derived_dir) first")
    for label, path in (("depth", DEPTH_DIR), ("rgb", RGB_DIR)):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{label} dataset directory does not exist: {path}")
    depth = sorted(glob.glob(os.path.join(DEPTH_DIR, "*.npy")))
    rgb = sorted(glob.glob(os.path.join(RGB_DIR, "*.png")))
    for label, paths in (("depth", depth), ("rgb", rgb)):
        if len(paths) < 2:
            raise ValueError(f"Need at least two {label} files for the training/validation split")
    return {"depth": depth, "rgb": rgb}


def prepare_occupancy_cache():
    depth = file_lists()["depth"]
    os.makedirs(OCC_CACHE, exist_ok=True)
    for i, f in enumerate(depth):
        out = os.path.join(OCC_CACHE, f"{i:06d}.npy")
        if not os.path.exists(out):
            np.save(out, bev_occupancy(np.load(f).astype(np.float32)))
        if i % 500 == 0:
            print(f"  occupancy cache {i}/{len(depth)}", flush=True)
    print(f"occupancy cache ready: {len(depth)} grids in {OCC_CACHE}")


# ---------------------------------------------------------------- loading
def _load_native(modality, files, i):
    """Native (nch, h, w) float32 in [0,1], before resize."""
    if modality == "depth":
        d = np.load(files["depth"][i]).astype(np.float32)
        d[~np.isfinite(d)] = 50.0
        return (np.clip(d, 0, 50) / 50.0)[None]
    if modality in ("occupancy", "costmap", "heatmap"):
        occ = np.load(os.path.join(OCC_CACHE, f"{i:06d}.npy"))
        if modality == "occupancy":
            g = occ
        elif modality == "costmap":
            g = inflate(occ)
        else:
            g = blur(inflate(occ)); g = g / (g.max() + 1e-9)
        return g[None].astype(np.float32)
    im = np.asarray(Image.open(files["rgb"][i]).convert("RGB"), np.float32) / 255.0   # (h,w,3)
    return np.ascontiguousarray(im.transpose(2, 0, 1))


def _resize(x, H, W):
    t = torch.from_numpy(x)[None]
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t[0]


def n_items(modality, files):
    return len(files["rgb"]) if modality == "rgb" else len(files["depth"])


def make_batch(modality, bs, H, W, files, idx_pool=None, pool=None):
    if idx_pool is None:
        idx_pool = range(n_items(modality, files))
    idx = [random.choice(idx_pool) for _ in range(bs)]
    load = lambda i: _resize(_load_native(modality, files, i), H, W)
    tensors = list(pool.map(load, idx)) if pool else [load(i) for i in idx]
    return torch.stack(tensors).float()                # (bs, nch, H, W)


def split_idx(modality, files, val_frac=0.1):
    n = n_items(modality, files); nv = max(1, int(val_frac * n))
    return list(range(0, n - nv)), list(range(n - nv, n))   # (train_idx, val_idx)


# ---------------------------------------------------------------- samplers
ASPECTS = [1.0, 4 / 3, 16 / 9, 3 / 4, 9 / 16]

def sample_size(lo=128, hi=768, mult=32):
    short = random.randrange(lo, hi + 1, mult)
    a = random.choice(ASPECTS)
    long = int(round(short * max(a, 1 / a) / mult)) * mult
    H, W = (short, long) if a >= 1 else (long, short)
    return max(mult, H), max(mult, W)

def batch_for(H, W, pixel_budget=1_600_000, bs_cap=16):
    return int(max(2, min(bs_cap, pixel_budget // (H * W))))   # keep GPU memory ~constant across sizes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--derived-dir", required=True)
    ap.add_argument("--prep", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not (a.prep or a.check):
        ap.error("choose --prep and/or --check")
    configure(a.depth_dir, a.rgb_dir, a.derived_dir)
    if a.prep:
        prepare_occupancy_cache()
    if a.check:
        files = file_lists()
        print("depth:", len(files["depth"]), " rgb:", len(files["rgb"]))
        for mod in list(GRID_MODS) + ["rgb"]:
            H, W = sample_size()
            b = make_batch(mod, batch_for(H, W), H, W, files)
            print(f"  {mod:10s} -> batch {tuple(b.shape)}  range [{b.min():.2f},{b.max():.2f}]")
