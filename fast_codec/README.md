# Fast codec — matched encoder + decoder, 4.8x faster end-to-end

`enc_fuse_stemdwpw4` encoder + `min_fused_t0` decoder, **trained together**.
Compression is unchanged from v2/v4: the latent is `C_lat = input channels` at
`1/2^depth` (verified 64.0x spatial at depth 3, both channel counts, depths 1-6).

> **These two halves are a matched pair.** Do not mix either with weights from
> `fast_decoders/` — those decoders were trained against the v4 encoder and the
> latents are not interchangeable.

## End-to-end result

GPU-resident, H200, depth 3, 5120x3840, FP16 + `torch.compile`. Quality is mean
PSNR over 5 modalities on held-out frames, 120 epochs x 600 steps.

| stack | encoder | decoder | **total** | 64x dB |
|---|---|---|---|---|
| original v4 | 2.169 ms | 0.226 ms | **2.395 ms** | 34.47 |
| **this** | **0.368 ms** | **0.133 ms** | **0.501 ms** | **34.98** |

**4.8x faster, slightly better quality.** The encoder supplied 1.80 ms of the
1.89 ms saved — 95% of the win — from the component that had never been touched
across v2, v4, or any prior work.

## Where the encoder time went

The v4 encoder's entire cost was its stem: `Conv(n->16, 3x3, stride 2)` at FULL
resolution. Measured cost is flat across exit depths (2.329 / 2.172 / 2.255 ms
at depths 1/3/6) because every later stage runs at a quarter the spatial extent
and is nearly free. That stem does 0.71 GMAC and moves ~196 MB against a
0.041 ms bandwidth roofline — **53x off its own roofline**. A 1-channel strided
convolution has poor arithmetic intensity and maps badly to tensor cores.

`enc_fuse_stemdwpw4` collapses the whole stem region into a single fused
pointwise kernel at 4 channels instead of 16, so the 157 MB half-resolution
16-channel intermediate is never materialised.

## What was tried and rejected

45 encoder variants were screened on a dedicated H200 (latency is
weight-independent, so no training is needed to rank them):

| approach | best result | verdict |
|---|---|---|
| fused narrow stem (`enc_fuse_*`) | **0.368 ms, -0.54 dB** | **shipped** |
| space-to-depth (`enc_s2d_*`) | 0.137 ms, -3.52 dB | 15.8x faster but too lossy |
| downsample reschedule (`enc_sched_*`) | 0.725 ms | mostly *slower* than baseline |

The quality/latency curve (clean H200 latency, 120-epoch quality):

| variant | ms | 64x dB |
|---|---|---|
| v4 baseline | 2.169 | 35.52 |
| enc_sched_narrow4 | 0.851 | 35.24 |
| **enc_fuse_stemdwpw4** | **0.368** | **34.98** |
| enc_s2d_u4 | 0.249 | 33.97 |
| enc_s2d_u8 | 0.137 | 32.00 |

Negative results worth keeping: `channels_last` changed nothing (2.172 vs
2.169); a 1x1 stride-2 stem was *slower* than 3x3 (2.303); writing the fused
stem as a reduction rather than unrolled FMAs was slower than baseline (2.213).
Within the fused family, **narrower was strictly better** — `stemdwpw4` beats
`stemdwpw8` and `stemdwpw16` on both latency and quality.

## Usage

```python
import sys; sys.path.insert(0, "fast_codec")
from load_example import load
enc, dec = load(device="cuda")
z = enc(x, depth)                 # x: (B,1|3,H,W) in [0,1]
y = dec(z, depth, x.shape[2], x.shape[3])
```

`python fast_codec/load_example.py` verifies the pair loads and the compression
contract holds.

## Caveats

- **Weights are final-epoch, not best-epoch** (the search harness validates only
  at the end; worth ~0.19 dB on the v4 reference run).
- **Latency is H200, not Orin.** See `fast_decoders/ORIN_BENCHMARK.md`. Kernel
  selection does not transfer: in this project a decoder family measured 1.8x
  *faster* than reference on A100 and 2.05x *slower* on H200, reproducibly.
- **Do not trust latency measured inside a training job.** On a contended node
  the same variant read up to 48x its dedicated-GPU value. All numbers here come
  from a dedicated-GPU screen.
- Validated on torch 2.5.1+cu121. Speedups hold in eager mode too (the decoder
  is 1.54x vs v4 without `torch.compile`), so a Triton backend is not required.
- **Quality is measured at 256x256 / 384x640 / 720x1280; latency at 5120x3840.**
  Those are different operating points; PSNR at the benchmark size is unverified.
