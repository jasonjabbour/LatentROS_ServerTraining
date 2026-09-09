#!/usr/bin/env python3
"""
train_multimodal.py — train the v2 codec across modalities, sizes, and rates in one model.

Each step samples a MODALITY (depth/occupancy/costmap/heatmap/rgb), a random SIZE (multi-scale, for size
generalization), and an exit-depth RATE (low-rate-weighted). A background thread prefetches batches so the
GPU stays busy. Loss = MSE. Validation reports PSNR per modality at three evaluation sizes.
Best-mean-PSNR checkpoints -> weights/.

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_multimodal.py --epochs 120
"""
import argparse
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

if __package__:
    from . import data_multimodal as D
    from .multimodal_ae import MMEncoder, MMDecoder, MAX_DEPTH
else:
    import data_multimodal as D
    from multimodal_ae import MMEncoder, MMDecoder, MAX_DEPTH

SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
MODS = ["depth", "occupancy", "costmap", "heatmap", "rgb"]
MOD_W = [2, 1, 1, 1, 2]                                # sample the real sensors (depth/rgb) a bit more
DEPTHS = list(range(1, MAX_DEPTH + 1))
DEPTH_W = [4, 3, 2, 1.5, 1, 1]                          # low (harder) rates more often
VAL_SIZES = [(256, 256), (384, 640), (720, 1280)]


def main(encoder_cls=MMEncoder, decoder_cls=MMDecoder, save=SAVE,
         default_tag="MMV2", default_steps=500):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--steps", type=int, default=default_steps, help="train steps per epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lo", type=int, default=128); ap.add_argument("--hi", type=int, default=768)
    ap.add_argument("--tag", default=default_tag)
    ap.add_argument("--output", default=save, help="checkpoint directory")
    ap.add_argument("--workers", type=int, default=8, help="parallel sample-loading threads")
    ap.add_argument("--prefetch", type=int, default=4, help="queued training batches")
    ap.add_argument("--pin-memory", action="store_true", help="pin prefetched batches for CUDA transfer")
    ap.add_argument("--torch-threads", type=int, help="CPU PyTorch threads; 1 avoids loader oversubscription")
    ap.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                    help="bf16 uses training autocast; validation stays fp32")
    ap.add_argument("--budget", type=int, default=1_600_000, help="batch pixel budget; changes effective batch")
    ap.add_argument("--batch-cap", type=int, default=16, help="maximum batch size; changes effective batch")
    a = ap.parse_args()
    if min(a.epochs, a.steps, a.workers, a.prefetch, a.budget, a.lo) < 1 or a.hi < a.lo or a.batch_cap < 2:
        ap.error("epochs, steps, workers, prefetch, budget and lo must be positive; hi >= lo; batch-cap >= 2")
    if a.torch_threads is not None:
        if a.torch_threads < 1:
            ap.error("torch-threads must be positive")
        torch.set_num_threads(a.torch_threads)
    dev = "cuda"
    files = D.file_lists()
    if any(len(files[m]) < 2 for m in ("depth", "rgb")):
        ap.error(f"need depth/*.npy and rgb/*.png with at least two files each under {D.DATA}")
    missing = next((i for i in range(len(files["depth"]))
                    if not os.path.isfile(os.path.join(D.OCC_CACHE, f"{i:06d}.npy"))), None)
    if missing is not None:
        ap.error(f"missing occupancy cache entry {missing:06d}; run {D.__file__} --prep")
    if not torch.cuda.is_available():
        ap.error("training requires a CUDA-enabled PyTorch installation and GPU")
    if a.precision == "bf16" and not torch.cuda.is_bf16_supported():
        ap.error("this GPU does not support bf16 training")
    tr_idx = {m: D.split_idx(m, files)[0] for m in MODS}
    va_idx = {m: D.split_idx(m, files)[1] for m in MODS}
    print(f"depth {len(files['depth'])}  rgb {len(files['rgb'])}  | train steps/epoch {a.steps}\n"
          f"precision {a.precision}; pixel budget {a.budget}; batch cap {a.batch_cap}; "
          f"loader workers {a.workers}; prefetch {a.prefetch}; pin memory {a.pin_memory}; "
          f"CPU torch threads {torch.get_num_threads()}", flush=True)

    enc = encoder_cls().to(dev); dec = decoder_cls().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    os.makedirs(a.output, exist_ok=True)
    pool = ThreadPoolExecutor(a.workers)

    # ---- background batch producer ----
    q = queue.Queue(maxsize=a.prefetch); stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def produce():
        try:
            while not stop.is_set():
                mod = random.choices(MODS, MOD_W)[0]
                H, W = D.sample_size(a.lo, a.hi)
                x = D.make_batch(mod, D.batch_for(H, W, a.budget, a.batch_cap), H, W, files, tr_idx[mod], pool)
                d = random.choices(DEPTHS, DEPTH_W)[0]
                put((mod, x.pin_memory() if a.pin_memory else x, d))
        except BaseException as exc:
            put(exc)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()

    best = -1.0
    try:
        for ep in range(1, a.epochs + 1):
            enc.train(); dec.train(); t0 = time.time(); tot = 0.0
            for _ in range(a.steps):
                item = q.get()
                if isinstance(item, BaseException):
                    raise RuntimeError("batch producer failed") from item
                mod, x, d = item
                x = x.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
                    xh = dec(enc(x, d), d, x.shape[2], x.shape[3])
                    loss = F.mse_loss(xh, x)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            sched.step()

            # ---- validation: PSNR per modality, averaged over held-out sizes+rates ----
            enc.eval(); dec.eval(); psnr = {}
            with torch.no_grad():
                for mod in MODS:
                    vals = []
                    for (H, W) in VAL_SIZES:
                        for d in (1, 3, 6):
                            x = D.make_batch(mod, 2, H, W, files, va_idx[mod], pool).to(dev)
                            xh = dec(enc(x, d), d, H, W)
                            mse = F.mse_loss(xh, x).item()
                            vals.append(10 * np.log10(1 / max(mse, 1e-9)))
                    psnr[mod] = float(np.mean(vals))
            mpsnr = float(np.mean(list(psnr.values())))
            line = "  ".join(f"{m[:4]}:{psnr[m]:.1f}" for m in MODS)
            print(f"ep {ep:3d}/{a.epochs}  {time.time()-t0:4.0f}s  train_mse {tot/a.steps:.5f}  "
                  f"PSNR[{line}]  mean {mpsnr:.2f} dB" + ("  *" if mpsnr > best else ""), flush=True)
            if mpsnr > best:
                best = mpsnr
                torch.save(enc.state_dict(), os.path.join(a.output, f"encoder_best_{a.tag}.pth"))
                torch.save(dec.state_dict(), os.path.join(a.output, f"decoder_best_{a.tag}.pth"))
    finally:
        stop.set()
        producer.join()
        pool.shutdown(wait=True)

    print(f"done. best mean-PSNR {best:.2f} dB -> {a.output}/*_{a.tag}.pth")


if __name__ == "__main__":
    main()
