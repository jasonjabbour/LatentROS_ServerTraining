#!/usr/bin/env python3
"""Loader variant of data_multimodal.py: npy RGB cache, native-resolution CPU
output, GPU-side normalisation and resize.

Sampling, derivation and value semantics are identical to the original. The
differences are mechanical: RGB comes from a pre-decoded uint8 cache instead of
PNG, and the per-sample cast/normalise/resize happens on the GPU in the trainer
rather than on the loader thread.
"""
import glob
import os
import random

import numpy as np
import torch

DEPTH_DIR = RGB_DIR = OCC_CACHE = None
GRID_MODS = ("depth", "occupancy", "costmap", "heatmap")
MOD_NCH = {"depth": 1, "occupancy": 1, "costmap": 1, "heatmap": 1, "rgb": 3}


# ---------------------------------------------------------------- derivation (unchanged)
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


def configure(depth_dir, rgb_dir, derived_dir):
    global DEPTH_DIR, RGB_DIR, OCC_CACHE
    DEPTH_DIR, RGB_DIR, OCC_CACHE = [os.path.abspath(os.path.expanduser(os.fspath(p)))
                                     for p in (depth_dir, rgb_dir, derived_dir)]


def file_lists():
    """RGB is read from a pre-decoded *.npy cache; depth unchanged."""
    if DEPTH_DIR is None:
        raise RuntimeError("Call configure(depth_dir, rgb_dir, derived_dir) first")
    for label, path in (("depth", DEPTH_DIR), ("rgb", RGB_DIR)):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{label} dataset directory does not exist: {path}")
    depth = sorted(glob.glob(os.path.join(DEPTH_DIR, "*.npy")))
    rgb = sorted(glob.glob(os.path.join(RGB_DIR, "*.npy")))
    if not rgb:
        raise FileNotFoundError(
            f"no *.npy in {RGB_DIR}; build the decoded RGB cache first (build_rgb_cache.py)")
    for label, paths in (("depth", depth), ("rgb", rgb)):
        if len(paths) < 2:
            raise ValueError(f"Need at least two {label} files for the training/validation split")
    return {"depth": depth, "rgb": rgb}


# ---------------------------------------------------------------- native CPU loading
def load_native(modality, files, i):
    """Native-resolution CPU tensor, NOT normalised. Normalisation happens on GPU.

    depth -> float32 (1,h,w) raw metres        (GPU: nan->50, clip 0..50, /50)
    rgb   -> uint8   (3,h,w) from npy cache    (GPU: float()/255)
    grids -> float32 (1,200,200) already [0,1] (GPU: no-op)
    """
    if modality == "depth":
        d = np.load(files["depth"][i])
        return torch.from_numpy(np.ascontiguousarray(d.astype(np.float32)))[None]
    if modality in ("occupancy", "costmap", "heatmap"):
        occ = np.load(os.path.join(OCC_CACHE, f"{i:06d}.npy"))
        if modality == "occupancy":
            g = occ
        elif modality == "costmap":
            g = inflate(occ)
        else:
            g = blur(inflate(occ)); g = g / (g.max() + 1e-9)
        return torch.from_numpy(np.ascontiguousarray(g.astype(np.float32)))[None]
    im = np.load(files["rgb"][i])                       # (h,w,3) uint8, lossless from PNG
    return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))


def normalise_gpu(modality, t):
    """Reproduce the original _load_native value semantics, on device."""
    if modality == "depth":
        t = torch.where(torch.isfinite(t), t, torch.full_like(t, 50.0))
        return t.clamp_(0.0, 50.0).div_(50.0)
    if modality == "rgb":
        return t.float().div_(255.0)
    return t


# ---------------------------------------------------------------- samplers (unchanged)
ASPECTS = [1.0, 4 / 3, 16 / 9, 3 / 4, 9 / 16]

def sample_size(lo=128, hi=768, mult=32):
    short = random.randrange(lo, hi + 1, mult)
    a = random.choice(ASPECTS)
    long = int(round(short * max(a, 1 / a) / mult)) * mult
    H, W = (short, long) if a >= 1 else (long, short)
    return max(mult, H), max(mult, W)

def batch_for(H, W, pixel_budget=1_600_000, bs_cap=16):
    return int(max(2, min(bs_cap, pixel_budget // (H * W))))

def n_items(modality, files):
    return len(files["rgb"]) if modality == "rgb" else len(files["depth"])

def split_idx(modality, files, val_frac=0.1):
    n = n_items(modality, files); nv = max(1, int(val_frac * n))
    return list(range(0, n - nv)), list(range(n - nv, n))
