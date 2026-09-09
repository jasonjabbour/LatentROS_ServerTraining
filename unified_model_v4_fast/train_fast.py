#!/usr/bin/env python3
"""Throughput variant of train_multimodal.py.

Same recipe (sampling distributions, loss, optimiser, schedule, validation).
Three mechanical changes to stop starving the GPU:
  1. RGB read from a pre-decoded uint8 npy cache instead of PNG-decoding.
  2. Cast/normalise/resize run on the GPU, not the loader.
  3. Batches are built by worker *processes*, sidestepping the GIL, instead of
     one producer thread driving a ThreadPoolExecutor.

Workers are forked before CUDA is initialised and never touch the GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

import data_fast as D
from multimodal_ae import MMEncoder, MMDecoder, MAX_DEPTH

HERE = Path(__file__).resolve().parent
MODS = ["depth", "occupancy", "costmap", "heatmap", "rgb"]
MOD_W = [2, 1, 1, 1, 2]
DEPTHS = list(range(1, MAX_DEPTH + 1))
DEPTH_W = [4, 3, 2, 1.5, 1, 1]
VAL_SIZES = [(256, 256), (384, 640), (720, 1280)]


def save_history(path, history):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as f:
        json.dump(history, f, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def pack(natives):
    """Flatten a list of native tensors into one buffer + shapes.

    RGB frames have mixed native sizes (1280x720, 1920x1080, 640x480), so a
    single stacked tensor is impossible; one flat buffer keeps it to a single
    shared-memory segment per batch instead of one per sample.
    """
    shapes = [tuple(t.shape) for t in natives]
    flat = torch.cat([t.reshape(-1) for t in natives])
    return flat, shapes


def unpack(flat, shapes):
    out = []
    off = 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(flat[off:off + n].view(*s))
        off += n
    return out


def producer(q, stop, seed, files, tr_idx, cfg):
    """Worker process: build one batch of native-resolution samples at a time."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        while not stop.is_set():
            mod = random.choices(MODS, MOD_W)[0]
            H, W = D.sample_size(cfg["lo"], cfg["hi"])
            bs = D.batch_for(H, W, cfg["budget"], cfg["batch_cap"])
            d = random.choices(DEPTHS, DEPTH_W)[0]
            idx = [random.choice(tr_idx[mod]) for _ in range(bs)]
            natives = [D.load_native(mod, files, i) for i in idx]
            flat, shapes = pack(natives)
            flat.share_memory_()
            while not stop.is_set():
                try:
                    q.put((mod, flat, shapes, d, H, W), timeout=0.1)
                    break
                except Exception:
                    continue
    except BaseException as exc:                     # surface to the trainer
        try:
            q.put(("__error__", repr(exc), None, None, None, None), timeout=5)
        except Exception:
            pass


def to_gpu_batch(mod, flat, shapes, H, W):
    """Normalise and resize on device, then stack. Mirrors _load_native+_resize."""
    flat = flat.cuda(non_blocking=True)
    outs = []
    off = 0
    for s in shapes:
        n = int(np.prod(s))
        t = flat[off:off + n].view(*s); off += n
        t = D.normalise_gpu(mod, t)
        t = F.interpolate(t[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        outs.append(t)
    return torch.stack(outs).float()


def val_batch(mod, bs, H, W, files, idx_pool):
    natives = [D.load_native(mod, files, random.choice(idx_pool)) for _ in range(bs)]
    flat, shapes = pack(natives)
    return to_gpu_batch(mod, flat, shapes, H, W)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth-dir", type=Path, required=True)
    ap.add_argument("--rgb-dir", type=Path, required=True, help="pre-decoded RGB *.npy cache")
    ap.add_argument("--derived-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lo", type=int, default=128); ap.add_argument("--hi", type=int, default=768)
    ap.add_argument("--tag", default="MMV4")
    ap.add_argument("--output", type=Path, default=HERE / "weights")
    ap.add_argument("--workers", type=int, default=8, help="batch-building worker PROCESSES")
    ap.add_argument("--prefetch", type=int, default=8, help="queued training batches")
    ap.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--budget", type=int, default=1_600_000)
    ap.add_argument("--batch-cap", type=int, default=16)
    a = ap.parse_args()
    if min(a.epochs, a.steps, a.workers, a.prefetch, a.budget, a.lo, a.lr) <= 0 or a.hi < a.lo or a.batch_cap < 2:
        ap.error("epochs, steps, workers, prefetch, budget, lo and lr must be positive; hi >= lo; batch-cap >= 2")
    if not a.tag or Path(a.tag).name != a.tag:
        ap.error("tag must be a filename suffix, without directory separators")
    for key in ("depth_dir", "rgb_dir", "derived_dir", "output"):
        setattr(a, key, getattr(a, key).expanduser().resolve())
    D.configure(a.depth_dir, a.rgb_dir, a.derived_dir)
    try:
        files = D.file_lists()
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    missing = next((i for i in range(len(files["depth"]))
                    if not (Path(D.OCC_CACHE) / f"{i:06d}.npy").is_file()), None)
    if missing is not None:
        ap.error(f"missing derived occupancy {missing:06d}.npy")
    if a.output.exists() and (not a.output.is_dir() or any(a.output.iterdir())):
        ap.error(f"output is not empty: {a.output}")

    tr_idx = {m: D.split_idx(m, files)[0] for m in MODS}
    va_idx = {m: D.split_idx(m, files)[1] for m in MODS}
    cfg = {"lo": a.lo, "hi": a.hi, "budget": a.budget, "batch_cap": a.batch_cap}

    # --- fork workers BEFORE touching CUDA; they only use numpy/torch-CPU
    ctx = mp.get_context("fork")
    q = ctx.Queue(maxsize=a.prefetch)
    stop = ctx.Event()
    procs = [ctx.Process(target=producer, args=(q, stop, 1234 + w, files, tr_idx, cfg), daemon=True)
             for w in range(a.workers)]
    for p in procs:
        p.start()

    # --- now CUDA
    if not torch.cuda.is_available():
        stop.set()
        ap.error("training requires a CUDA-enabled PyTorch installation and GPU")
    if a.precision == "bf16" and not torch.cuda.is_bf16_supported():
        stop.set()
        ap.error("this GPU does not support bf16 training")
    print(f"depth {len(files['depth'])}  rgb {len(files['rgb'])}  derived {D.OCC_CACHE}\n"
          f"steps/epoch {a.steps}; precision {a.precision}; pixel budget {a.budget}; batch cap {a.batch_cap}; "
          f"worker PROCESSES {a.workers}; prefetch {a.prefetch}; gpu resize ON; rgb npy cache ON", flush=True)

    enc = MMEncoder().cuda(); dec = MMDecoder().cuda()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    props = torch.cuda.get_device_properties(0)
    history = {
        "config": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                   "variant": "fast: npy-rgb + gpu-resize + process-workers",
                   "modality_weights": dict(zip(MODS, MOD_W)), "exit_weights": dict(zip(DEPTHS, DEPTH_W)),
                   "validation_sizes": VAL_SIZES, "validation_exits": [1, 3, 6], "validation_batch": 2,
                   "train_counts": {m: len(v) for m, v in tr_idx.items()},
                   "validation_counts": {m: len(v) for m, v in va_idx.items()},
                   "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                     for name in ("train_fast.py", "data_fast.py", "multimodal_ae.py")}},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "gpu": props.name, "gpu_memory_bytes": props.total_memory,
                    "gpu_capability": [props.major, props.minor]},
        "epochs": [], "best_epoch": None, "best_psnr_db": None, "completed": False,
    }
    a.output.mkdir(parents=True, exist_ok=True)
    history_path = a.output / "history.json"
    save_history(history_path, history)

    best = -1.0
    try:
        for ep in range(1, a.epochs + 1):
            enc.train(); dec.train(); t0 = time.perf_counter(); tot = 0.0
            lr_used = opt.param_groups[0]["lr"]
            train_counts = dict.fromkeys(MODS, 0)
            for _ in range(a.steps):
                mod, flat, shapes, d, H, W = q.get()
                if mod == "__error__":
                    raise RuntimeError(f"batch producer failed: {flat}")
                train_counts[mod] += 1
                x = to_gpu_batch(mod, flat, shapes, H, W)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
                    xh = dec(enc(x, d), d, x.shape[2], x.shape[3])
                    loss = F.mse_loss(xh, x)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            sched.step()

            enc.eval(); dec.eval(); psnr = {}; val_mse = {}
            with torch.no_grad():
                for mod in MODS:
                    vals = []; mses = []
                    for H, W in VAL_SIZES:
                        for d in (1, 3, 6):
                            x = val_batch(mod, 2, H, W, files, va_idx[mod])
                            xh = dec(enc(x, d), d, H, W)
                            mse = F.mse_loss(xh, x).item()
                            mses.append(mse); vals.append(10 * np.log10(1 / max(mse, 1e-9)))
                    psnr[mod] = float(np.mean(vals)); val_mse[mod] = float(np.mean(mses))
            mpsnr = float(np.mean(list(psnr.values())))
            line = "  ".join(f"{m[:4]}:{psnr[m]:.1f}" for m in MODS)
            elapsed = time.perf_counter() - t0
            print(f"ep {ep:3d}/{a.epochs}  {elapsed:4.0f}s  train_mse {tot/a.steps:.5f}  "
                  f"PSNR[{line}]  mean {mpsnr:.2f} dB" + ("  *" if mpsnr > best else ""), flush=True)
            if mpsnr > best:
                best = mpsnr
                torch.save(enc.state_dict(), a.output / f"encoder_best_{a.tag}.pth")
                torch.save(dec.state_dict(), a.output / f"decoder_best_{a.tag}.pth")
                history.update(best_epoch=ep, best_psnr_db=best)
            history["epochs"].append({"epoch": ep, "train_mse": tot / a.steps,
                                      "train_steps_by_modality": train_counts,
                                      "validation_mse": {**val_mse, "mean": float(np.mean(list(val_mse.values())))},
                                      "validation_psnr_db": {**psnr, "mean": mpsnr},
                                      "learning_rate": lr_used, "seconds": elapsed})
            history["completed"] = ep == a.epochs
            save_history(history_path, history)
    finally:
        stop.set()
        try:
            while not q.empty():
                q.get_nowait()
        except Exception:
            pass
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    print(f"done. best mean-PSNR {best:.2f} dB -> {a.output}")


if __name__ == "__main__":
    main()
