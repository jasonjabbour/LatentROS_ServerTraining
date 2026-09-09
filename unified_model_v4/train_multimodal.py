#!/usr/bin/env python3
"""Train both v4 networks from scratch using the v2 sampling/loss recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
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

HERE = Path(__file__).resolve().parent
MODS = ["depth", "occupancy", "costmap", "heatmap", "rgb"]
MOD_W = [2, 1, 1, 1, 2]
DEPTHS = list(range(1, MAX_DEPTH + 1))
DEPTH_W = [4, 3, 2, 1.5, 1, 1]
VAL_SIZES = [(256, 256), (384, 640), (720, 1280)]


def save_history(path, history):
    """Keep the previous complete epoch if writing the next one is interrupted."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as f:
        json.dump(history, f, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth-dir", type=Path, required=True, help="directory containing depth *.npy frames")
    ap.add_argument("--rgb-dir", type=Path, required=True, help="directory containing RGB *.png frames")
    ap.add_argument("--derived-dir", type=Path, required=True, help="prepared occupancy cache directory")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--steps", type=int, default=600, help="training updates per epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lo", type=int, default=128); ap.add_argument("--hi", type=int, default=768)
    ap.add_argument("--tag", default="MMV4", help="checkpoint filename suffix")
    ap.add_argument("--output", type=Path, default=HERE / "weights", help="new or empty directory for weights and history.json")
    ap.add_argument("--workers", type=int, default=8, help="parallel sample-loading threads")
    ap.add_argument("--prefetch", type=int, default=4, help="queued training batches")
    ap.add_argument("--pin-memory", action="store_true", help="pin prefetched batches for CUDA transfer")
    ap.add_argument("--torch-threads", type=int, help="CPU PyTorch threads; 1 avoids loader oversubscription")
    ap.add_argument("--precision", choices=("fp32", "bf16"), default="fp32", help="bf16 training autocast; validation stays fp32")
    ap.add_argument("--budget", type=int, default=1_600_000, help="batch pixel budget; changes effective batch")
    ap.add_argument("--batch-cap", type=int, default=16, help="maximum batch size; changes effective batch")
    a = ap.parse_args()
    if min(a.epochs, a.steps, a.workers, a.prefetch, a.budget, a.lo, a.lr) <= 0 or a.hi < a.lo or a.batch_cap < 2:
        ap.error("epochs, steps, workers, prefetch, budget, lo and lr must be positive; hi >= lo; batch-cap >= 2")
    if not a.tag or Path(a.tag).name != a.tag:
        ap.error("tag must be a filename suffix, without directory separators")
    if a.torch_threads is not None:
        if a.torch_threads < 1:
            ap.error("torch-threads must be positive")
        torch.set_num_threads(a.torch_threads)
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
        ap.error(f"missing derived occupancy {missing:06d}.npy; run data_multimodal.py --prep with the same paths")
    if a.output.exists() and (not a.output.is_dir() or any(a.output.iterdir())):
        ap.error(f"output is not empty: {a.output}; choose a new --output to preserve earlier results")
    if not torch.cuda.is_available():
        ap.error("training requires a CUDA-enabled PyTorch installation and GPU")
    if a.precision == "bf16" and not torch.cuda.is_bf16_supported():
        ap.error("this GPU does not support bf16 training")
    tr_idx = {m: D.split_idx(m, files)[0] for m in MODS}
    va_idx = {m: D.split_idx(m, files)[1] for m in MODS}
    print(f"depth {len(files['depth'])}  rgb {len(files['rgb'])}  derived {D.OCC_CACHE}\n"
          f"steps/epoch {a.steps}; precision {a.precision}; pixel budget {a.budget}; batch cap {a.batch_cap}; "
          f"loader workers {a.workers}; prefetch {a.prefetch}; pin memory {a.pin_memory}; "
          f"CPU torch threads {torch.get_num_threads()}", flush=True)

    enc = MMEncoder().cuda(); dec = MMDecoder().cuda()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    props = torch.cuda.get_device_properties(0)
    history = {
        "config": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                   "modality_weights": dict(zip(MODS, MOD_W)), "exit_weights": dict(zip(DEPTHS, DEPTH_W)),
                   "validation_sizes": VAL_SIZES, "validation_exits": [1, 3, 6], "validation_batch": 2,
                   "train_counts": {m: len(v) for m, v in tr_idx.items()},
                   "validation_counts": {m: len(v) for m, v in va_idx.items()},
                   "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                     for name in ("train_multimodal.py", "data_multimodal.py", "multimodal_ae.py")}},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "gpu": props.name, "gpu_memory_bytes": props.total_memory,
                    "gpu_capability": [props.major, props.minor], "torch_threads": torch.get_num_threads(),
                    "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32},
        "epochs": [], "best_epoch": None, "best_psnr_db": None, "completed": False,
    }
    a.output.mkdir(parents=True, exist_ok=True)
    history_path = a.output / "history.json"
    save_history(history_path, history)
    pool = ThreadPoolExecutor(a.workers)
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
            enc.train(); dec.train(); t0 = time.perf_counter(); tot = 0.0
            lr_used = opt.param_groups[0]["lr"]
            train_counts = dict.fromkeys(MODS, 0)
            for _ in range(a.steps):
                item = q.get()
                if isinstance(item, BaseException):
                    raise RuntimeError("batch producer failed") from item
                mod, x, d = item
                train_counts[mod] += 1
                x = x.cuda(non_blocking=True)
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
                            x = D.make_batch(mod, 2, H, W, files, va_idx[mod], pool).cuda()
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
        producer.join()
        pool.shutdown(wait=True)
    print(f"done. best mean-PSNR {best:.2f} dB -> {a.output}")

if __name__ == "__main__":
    main()
