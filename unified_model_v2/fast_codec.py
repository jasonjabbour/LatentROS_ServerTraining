#!/usr/bin/env python3
"""
fast_codec.py — ROS-ready fast inference wrapper for the unified codec (v2), with quality-preserving system
optimizations, each individually toggleable:

  * fp16            — half precision (verified bit-identical PSNR).
  * torch.compile   — fuses conv/pixelshuffle/refine into optimized kernels (~3x on the decoder).
  * dynamic shapes  — compile ONCE for dynamic H/W so a GROWING / changing message size does NOT trigger a
                      slow recompile each time (the key requirement for variable-size ROS messages).
  * pinned memory   — messages come from CPU; a pinned host buffer + non_blocking copy speeds the H->D transfer.
  * channels_last   — alternate memory format (usually a wash here; left as a switch).

None of these change the math (fp16 is the only precision change, and it's verified). Use `--verify` to
confirm the fast path matches fp32 before trusting it.

  python fast_codec.py --bench                    # growing-message latency sweep (default: fp16+compile+dynamic+pin)
  python fast_codec.py --bench --no-compile        # toggle any optimization off
  python fast_codec.py --bench --static            # compile per-shape (shows the recompile stalls dynamic avoids)
  python fast_codec.py --verify                    # fast path vs fp32 fidelity
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


class FastCodec:
    def __init__(self, weights=None, fp16=True, compile=True, dynamic=True, mode=None,
                 pin=True, channels_last=False, device="cuda"):
        sys.path.insert(0, HERE)
        from multimodal_ae import MMEncoder, MMDecoder
        weights = weights or os.path.join(HERE, "weights")
        self.dev, self.fp16, self.pin, self.cl = device, fp16, pin, channels_last
        self.dtype = torch.half if fp16 else torch.float
        self.opts = dict(fp16=fp16, compile=compile, dynamic=dynamic, mode=mode, pin=pin, channels_last=channels_last)
        enc, dec = MMEncoder().to(device).eval(), MMDecoder().to(device).eval()
        enc.load_state_dict(torch.load(f"{weights}/encoder_best_MMV2.pth", map_location=device))
        dec.load_state_dict(torch.load(f"{weights}/decoder_best_MMV2.pth", map_location=device))
        if fp16:
            enc.half(); dec.half()
        if channels_last:
            enc = enc.to(memory_format=torch.channels_last); dec = dec.to(memory_format=torch.channels_last)
        self._enc_raw, self._dec_raw = enc, dec                 # (fp16/channels_last applied) raw modules
        self.dynamic, self.compile = dynamic, compile
        self._pin_buf = None                                    # reusable pinned host buffer (grown as needed)
        self._ckw = {"mode": mode} if mode else {}
        self._enc_c, self._dec_c = {}, {}                       # per-DEPTH compiled fns: depth is baked in as a
        #   constant (so `1 << depth` etc. never go symbolic), while H/W are marked dynamic per call.
        if compile:
            # respect ONLY our explicit mark_dynamic(H,W); don't auto-promote depth to a symbol on shape change
            torch._dynamo.config.automatic_dynamic_shapes = False

    def _enc_fn(self, depth):
        if depth not in self._enc_c:
            fn = (lambda t: self._enc_raw(t, depth))            # depth CAPTURED (constant), not a param
            self._enc_c[depth] = torch.compile(fn, **self._ckw) if self.compile else fn
        return self._enc_c[depth]

    def _dec_fn(self, depth):
        if depth not in self._dec_c:
            fn = (lambda z: self._dec_raw(z, depth))            # captured depth; no crop inside -> shape-agnostic
            self._dec_c[depth] = torch.compile(fn, **self._ckw) if self.compile else fn
        return self._dec_c[depth]

    def _to_gpu(self, grid):
        a = np.ascontiguousarray(grid[None] if grid.ndim == 2 else grid, np.float32)   # (nch,H,W)
        if self.pin:                                              # copy into a REUSED pinned buffer (fast H->D)
            n = a.size
            if self._pin_buf is None or self._pin_buf.numel() < n:
                self._pin_buf = torch.empty(max(n, 1), dtype=torch.float32).pin_memory()
            host = self._pin_buf[:n].view(a.shape); host.copy_(torch.from_numpy(a))
            t = host.to(self.dev, non_blocking=True)
        else:
            t = torch.from_numpy(a).to(self.dev, non_blocking=True)
        t = t.to(self.dtype).unsqueeze(0)                         # (1,nch,H,W)
        if self.cl:
            t = t.contiguous(memory_format=torch.channels_last)
        return t

    @torch.no_grad()
    def encode(self, grid, depth):
        """grid: numpy (nch,H,W) or (H,W) on CPU -> latent tensor on GPU."""
        x = self._to_gpu(grid)
        if self.dynamic:
            torch._dynamo.mark_dynamic(x, 2); torch._dynamo.mark_dynamic(x, 3)   # H,W dynamic; depth stays static
        return self._enc_fn(depth)(x)

    @torch.no_grad()
    def decode(self, latent, depth, H, W):
        """latent (GPU) -> reconstructed numpy (nch,H,W) on CPU."""
        if self.dynamic:
            torch._dynamo.mark_dynamic(latent, 2); torch._dynamo.mark_dynamic(latent, 3)
        out = self._dec_fn(depth)(latent)                      # crop OUTSIDE compile -> shape-agnostic kernel
        out = out[:, :, :H, :W].clamp(0, 1)[0]                 # keep model dtype (fp16) for the D->H copy...
        return out.cpu().numpy().astype(np.float32)            # ...half the bytes, then widen on the CPU side

    @torch.no_grad()
    def roundtrip(self, grid, depth):
        H, W = grid.shape[-2:]
        return self.decode(self.encode(grid, depth), depth, H, W)

    @torch.no_grad()
    def compute_only(self, x_gpu, depth, H, W):
        """enc+dec on an already-on-GPU tensor, output kept on GPU — pure codec speed, no CPU transfer."""
        if self.dynamic:
            torch._dynamo.mark_dynamic(x_gpu, 2); torch._dynamo.mark_dynamic(x_gpu, 3)
        z = self._enc_fn(depth)(x_gpu)
        return self._dec_fn(depth)(z)[:, :, :H, :W]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(codec, sizes, depth=2, nch=1, warm=6, reps=10):
    print(f"opts: {codec.opts}   depth {depth} ({4**depth}x)\n")
    print(f"{'size':>11} | {'1st call':>8} | {'compute (on-GPU)':>16} | {'end-to-end (from CPU)':>21}")
    print(f"{'':>11} | {'ms':>8} | {'enc+dec ms':>16} | {'enc / dec / total ms':>21}")
    print("-" * 66)
    for (H, W) in sizes:
        g = np.random.rand(nch, H, W).astype(np.float32)
        try:
            t0 = time.perf_counter(); codec.roundtrip(g, depth); _sync()
            first = (time.perf_counter() - t0) * 1000                        # includes any (re)compile
            for _ in range(warm):
                codec.roundtrip(g, depth)
            x = codec._to_gpu(g)                                  # compute-only: input already on GPU
            _sync(); s = time.perf_counter()
            for _ in range(reps):
                codec.compute_only(x, depth, H, W)
            _sync(); comp = (time.perf_counter() - s) / reps * 1000
            _sync(); s = time.perf_counter()                     # end-to-end encode (incl H->D)
            for _ in range(reps):
                z = codec.encode(g, depth)
            _sync(); e = (time.perf_counter() - s) / reps * 1000
            _sync(); s = time.perf_counter()                     # end-to-end decode (incl D->H)
            for _ in range(reps):
                codec.decode(z, depth, H, W)
            _sync(); d = (time.perf_counter() - s) / reps * 1000
            print(f"{W:>5}x{H:<5} | {first:>8.1f} | {comp:>16.2f} | {e:>5.2f} / {d:>5.2f} / {e+d:>6.2f}")
        except RuntimeError as ex:
            print(f"{W:>5}x{H:<5} | OOM/err ({str(ex)[:40]})")
        finally:
            torch.cuda.empty_cache()
    print("\n1st-call ~= steady state -> NO recompile stall on that new size (the point of dynamic compile).")
    print("compute = pure codec on GPU; end-to-end adds the CPU<->GPU copies of the full grid.")


def verify(depth=2, size=(768, 512)):
    fp32 = FastCodec(fp16=False, compile=False, pin=False)
    fast = FastCodec(fp16=True, compile=True, dynamic=True, pin=True)
    H, W = size
    for nch, name in [(1, "grid"), (3, "rgb")]:
        g = np.random.rand(nch, H, W).astype(np.float32)
        a = fp32.roundtrip(g, depth); b = fast.roundtrip(g, depth)
        psnr_ab = 10 * np.log10(1 / max(((a - b) ** 2).mean(), 1e-12))
        print(f"  {name}: fp32 vs fast reconstruction match = {psnr_ab:.1f} dB (higher=identical; >50 dB is lossless-grade)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--static", action="store_true", help="compile per-shape instead of dynamic")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--nch", type=int, default=1)
    a = ap.parse_args()
    if a.verify:
        verify(a.depth)
    if a.bench:
        codec = FastCodec(fp16=not a.no_fp16, compile=not a.no_compile, dynamic=not a.static,
                          mode=a.mode, pin=not a.no_pin, channels_last=a.channels_last)
        SIZES = [(512, 512), (1024, 1024), (1536, 1536), (2048, 2048), (3000, 3000), (4000, 4000)]  # growing msg
        benchmark(codec, SIZES, depth=a.depth, nch=a.nch)
