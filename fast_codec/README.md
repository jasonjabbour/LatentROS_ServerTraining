# Fast codec — matched encoder + decoder, 5.4x faster end-to-end

`enc_s2d_u8_x2` encoder + `min_fused_t0` decoder, **trained together**.
Compression is unchanged from v2/v4: the latent is `C_lat = input channels` at
`1/2^depth` (verified 64.0x spatial at depth 3, both channel counts, depths 1-6,
including non-divisible sizes).

> **These two halves are a matched pair.** Do not mix either with weights from
> `fast_decoders/` — those decoders were trained against the v4 encoder and the
> latents are not interchangeable.

## End-to-end result

GPU-resident, H200, depth 3, 5120x3840, FP16 + `torch.compile`. Quality is mean
PSNR over 5 modalities on held-out frames, 120 epochs x 600 steps, identical
harness and split for every row.

| stack | encoder | decoder | **total** | 4x | 16x | 64x | 256x |
|---|---|---|---|---|---|---|---|
| original v4 | 2.170 ms | 0.226 ms | **2.396 ms** | 39.93 | 38.43 | 35.52 | 31.60 |
| **this** | **0.307 ms** | **0.133 ms** | **0.440 ms** | **41.98** | **40.22** | **35.85** | **32.26** |

**5.4x faster, and higher PSNR at all four compression rates.** The encoder
supplied 1.86 ms of the 1.96 ms saved — 95% of the win — and drops from 94% to
70% of total codec GPU compute.

Cross-GPU: measured on A100 as well, where a *different* cuDNN pathology
dominates. This encoder reads **0.659 ms against a 34.83 ms baseline — 52.8x** —
so the win is not an H200-specific artifact.

| encoder | H200 ms | A100 ms |
|---|---|---|
| v4 baseline | 2.170 | 34.83 |
| this (`enc_s2d_u8_x2`) | **0.307** | **0.659** |

Per-modality at 64x: heatmap 46.77, costmap 42.82, occupancy 34.23,
depth 30.10, rgb 25.34.

## Why the encoder was slow, and why this is fast

v4's entire encoder cost was its stem, `Conv(n->16, 3x3, stride 2)` at FULL
resolution — cost is flat across exit depths (2.329 / 2.172 / 2.255 ms at depths
1/3/6) because every later stage runs at a quarter the spatial extent. Profiling
shows **83% of the 2.17 ms is a single kernel**: cuDNN treats a 1-input-channel
convolution as a *grouped* conv and selects `conv2d_grouped_direct_kernel`, a
direct non-tensor-core path.

`PixelUnshuffle(8)` fixes both problems at once:
- the stem conv becomes `Conv1x1(64n -> 64)` — **tensor-core-shaped**, no longer
  `Cin=1`. Unshuffle(2)+Conv1x1 alone recovers 3.5x at the *same* 157 MB write.
- the dominant write moves from **157 MB at H/2 to 39 MB at H/8**. Latency tracks
  stem output bytes near-linearly: 157 MB -> 0.63 ms, 79 MB -> 0.25, 39 MB -> 0.13.

Unshuffle alone (`u8`, no extra convs) is a single linear map of each disjoint
8x8 block and loses 3.4 dB. **Two 3x3 convs at H/8 restore cross-block mixing for
~0.17 ms and buy back +3.71 dB** — capacity at reduced resolution is the cheapest
quality in this design space.

## What was tried and rejected

73 encoder variants across three hypotheses, screened on a dedicated H200
(latency is weight-independent, so ranking needs no training):

| approach | best | verdict |
|---|---|---|
| space-to-depth (`enc_s2d_*`) | **0.307 ms, +0.33 dB** | **shipped** |
| fused narrow stem (`enc_fuse_*`) | 0.304 ms, -0.34 dB | ties on latency, ~0.9 dB worse |
| downsample reschedule (`enc_sched_*`) | 0.848 ms, -0.29 dB | mostly *slower* than baseline |

Negative results worth not repeating:
- **`channels_last` did nothing** (2.164 vs 2.164).
- **A 1x1 stride-2 stem is *slower* than 3x3** (1.32 vs 2.17 is the fused case;
  the naive 1x1 stem gains little) — FLOPs are not the cost.
- **`ConvTranspose2d`-style and reduction formulations are worse** than the
  algebraically identical alternatives; inductor materialises a large
  intermediate instead of fusing.
- **The decoder's fused-tail trick inverts here**: unrolled broadcast math helped
  at unshuffle 2 (1.5x) but was **10x worse** at unshuffle 8 — 4 unrolled terms
  fuse well, 64 terms generate a losing pointwise loop.
- **Widening low-resolution stages is not free at scale** (+0.35 ms for width 256).
- **Half resolution is not free either**: two extra 16->16 3x3 convs at H/2 cost
  +1.6 ms. Only H/4 and below is cheap.

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

## Caveats — read before quoting the quality numbers

- **The quality margin is single-seed.** The harness seeds data order but **not**
  model init, and the same architecture elsewhere in this search moved **0.52 dB**
  between runs (and 2.16 dB at 4x for two identical depth-1 paths). The +0.33 dB
  at 64x is inside that range. Being higher at all four rates simultaneously is
  suggestive, not proof. **The latency win is not marginal and needs no hedging;
  the quality claim wants 2-3 seeds before it goes in a paper.**
- **Weights are final-epoch, not best-epoch** (worth ~0.19 dB on the v4 reference).
- **Latency is H200/A100, not Orin.** See `fast_decoders/ORIN_BENCHMARK.md`.
  Kernel selection does not transfer *at all*: the A100 encoder baseline is
  **34.8 ms vs H200's 2.17 ms** — 16x, not a scaling factor — because a different
  cuDNN kernel dominates there, and rankings invert between the two.
- **Do not screen encoder work on A100.** Every unshuffle-2 variant read ~34 ms
  there, making the stem fix completely invisible.
- **Quality is measured at 256x256 / 384x640 / 720x1280; latency at 5120x3840.**
  Different operating points; PSNR at the benchmark size is unverified.
- Validated on torch 2.5.1+cu121. Some variants required raising inductor's
  `realize_opcount_threshold` / `realize_reads_threshold`; this one does not.
