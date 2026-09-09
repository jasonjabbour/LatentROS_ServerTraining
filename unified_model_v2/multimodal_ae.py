#!/usr/bin/env python3
"""
multimodal_ae.py — LatentROS unified codec v2: ONE model, variable rate, any input size, any 2D-grid modality.

Extends the v1 early-exit design (rate = spatial exit depth d, ratio 4^d, one shared trunk) with:
  * MULTI-MODALITY — per-channel-count input stems and output heads (1-ch for depth/occupancy/costmap/heatmap,
    3-ch for RGB) around a fully SHARED trunk + shared upsampler. Only the thin end layers differ per modality.
  * MATCHED LATENT — the latent has C_lat = in_ch channels, so every modality gets ratio 4^d at exit depth d
    (1-ch latent for grids, 3-ch for RGB — enough capacity for colour).
  * ANY SIZE — fully convolutional (reflect-pad to a multiple of 2^depth); size generalization comes from
    multi-scale training (see train_multimodal.py), not the architecture.

  python multimodal_ae.py --bench      # per-rate, per-modality latency + shapes
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTHS = [16, 32, 64, 128, 128, 128]              # encoder stage output width (depth d -> WIDTHS[d-1])
DEC_WIDTHS = [32, 32, 48, 64, 128, 128, 128]      # decoder width per level /2^L (wide at full-res for fidelity)
MAX_DEPTH = len(WIDTHS)
NCHS = (1, 3)                                      # supported input/output channel counts (grids=1, RGB=3)


class DWSep(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(ic, ic, 3, stride, 1, groups=ic, bias=False)
        self.pw = nn.Conv2d(ic, oc, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class MMEncoder(nn.Module):
    def __init__(self, max_depth=MAX_DEPTH):
        super().__init__()
        self.max_depth = max_depth
        # per-channel-count stem (this is E1, the full-res downsample)
        self.stem = nn.ModuleDict({str(n): nn.Sequential(nn.Conv2d(n, WIDTHS[0], 3, 2, 1), nn.ReLU(inplace=True))
                                   for n in NCHS})
        # shared stages E2..E_MAX
        self.stages = nn.ModuleList([DWSep(WIDTHS[d - 2], WIDTHS[d - 1], stride=2) for d in range(2, max_depth + 1)])
        # per-channel latent head at each depth: WIDTHS[d-1] -> n  (matched latent, C_lat = in_ch)
        self.to_latent = nn.ModuleDict({str(n): nn.ModuleList([nn.Conv2d(WIDTHS[d - 1], n, 1)
                                                               for d in range(1, max_depth + 1)]) for n in NCHS})

    def forward(self, x, depth):
        n = x.shape[1]
        h, w = x.shape[-2:]
        s = 1 << depth
        ph, pw = (s - h % s) % s, (s - w % s) % s
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        x = self.stem[str(n)](x)                       # E1
        for i in range(depth - 1):                     # E2..E_depth (shared)
            x = self.stages[i](x)
        return self.to_latent[str(n)][depth - 1](x)    # (B, n, H/2^depth, W/2^depth)


class MMDecoder(nn.Module):
    def __init__(self, max_depth=MAX_DEPTH):
        super().__init__()
        self.max_depth = max_depth
        self.from_latent = nn.ModuleDict({str(n): nn.ModuleList(
            [nn.Sequential(nn.Conv2d(n, DEC_WIDTHS[d], 1), nn.ReLU(inplace=True)) for d in range(1, max_depth + 1)])
            for n in NCHS})
        # shared upsample stages U[d]: DEC_WIDTHS[d] -> DEC_WIDTHS[d-1] at x2
        self.ups = nn.ModuleList([nn.Sequential(DWSep(DEC_WIDTHS[d], DEC_WIDTHS[d - 1] * 4),
                                                nn.PixelShuffle(2), nn.ReLU(inplace=True))
                                  for d in range(1, max_depth + 1)])
        self.refine = nn.ModuleDict({str(n): nn.Sequential(nn.Conv2d(DEC_WIDTHS[0], n, 3, padding=1), nn.Sigmoid())
                                     for n in NCHS})

    def forward(self, z, depth, th=None, tw=None):
        n = z.shape[1]
        x = self.from_latent[str(n)][depth - 1](z)
        for d in range(depth, 0, -1):
            x = self.ups[d - 1](x)
        x = self.refine[str(n)](x)
        if th and tw and (x.shape[2] != th or x.shape[3] != tw):
            x = x[:, :, :th, :tw]
        return x


def _time(fn, warmup=15, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def bench():
    dev = 'cuda'
    enc = MMEncoder().to(dev).eval(); dec = MMDecoder().to(dev).eval()
    nparams = sum(p.numel() for m in (enc, dec) for p in m.parameters())
    print(f"one model: {nparams:,} params on {torch.cuda.get_device_name(0)}\n")
    with torch.no_grad():
        for nch, (H, W), label in [(1, (720, 1280), 'grid  720x1280'), (3, (720, 1280), 'rgb   720x1280'),
                                   (3, (480, 640), 'rgb   480x640'), (1, (200, 200), 'grid  200x200')]:
            x = torch.rand(1, nch, H, W, device=dev)
            print(f"{label}:")
            for d in (1, 3, 6):
                z = enc(x, d)
                te = _time(lambda: enc(x, d)); td = _time(lambda: dec(z, d, H, W))
                ratio = (nch * H * W) / z.numel()
                print(f"   rate {ratio:6.0f}x (depth {d})  latent {tuple(z.shape[1:])}  enc {te:.2f} ms  dec {td:.2f} ms")
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bench", action="store_true"); a = ap.parse_args()
    if a.bench:
        bench()
