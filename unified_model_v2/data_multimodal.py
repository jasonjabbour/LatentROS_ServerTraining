#!/usr/bin/env python3
"""
data_multimodal.py — multi-modality, multi-scale data for the v2 codec.

Modalities: depth (CARLA .npy), rgb (collected .png), and occupancy/costmap/heatmap DERIVED from the depth
frames (back-project -> BEV occupancy -> inflate -> smooth). Occupancy is cached once (fast); cost-map and
heat-map are derived from the cache on the fly (cheap). Every sample is resized to a random (H, W) per batch
for size generalization. All values are normalized to [0, 1]; grids are 1-channel, RGB is 3-channel.

  python data_multimodal.py --prep      # build the occupancy cache (one time, ~1-2 min)
  python data_multimodal.py --check      # sanity: load one batch of every modality
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "datasets", "dataset")
DEPTH_DIR = os.path.join(DATA, "depth")
RGB_DIR = os.path.join(DATA, "rgb")
OCC_CACHE = os.path.join(HERE, "derived_occ")          # cached 200x200 BEV occupancy, one per depth frame
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
def file_lists():
    depth = sorted(glob.glob(os.path.join(DEPTH_DIR, "*.npy")))
    rgb = sorted(glob.glob(os.path.join(RGB_DIR, "*.png")))
    return {"depth": depth, "rgb": rgb}


def prepare_occupancy_cache():
    os.makedirs(OCC_CACHE, exist_ok=True)
    depth = sorted(glob.glob(os.path.join(DEPTH_DIR, "*.npy")))
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
    ap.add_argument("--prep", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.prep:
        prepare_occupancy_cache()
    if a.check:
        files = file_lists()
        print("depth:", len(files["depth"]), " rgb:", len(files["rgb"]))
        for mod in list(GRID_MODS) + ["rgb"]:
            H, W = sample_size()
            b = make_batch(mod, batch_for(H, W), H, W, files)
            print(f"  {mod:10s} -> batch {tuple(b.shape)}  range [{b.min():.2f},{b.max():.2f}]")
