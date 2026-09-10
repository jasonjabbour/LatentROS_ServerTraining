# Fast decoders — same compression as v2/v4, lower latency

> **Superseded by `unified_model_v5/`**, which rebuilds the encoder too and is
> 5.4x faster end-to-end (0.440 ms vs v4's 2.396 ms) at higher PSNR on all four
> rates. The decoders here were trained against the **v4 encoder** and remain
> valid as v4-compatible drop-ins; do not pair their weights with v5's encoder.
> `ORIN_BENCHMARK.md` in this directory applies to both.

Two trained decoders that keep the codec's exact compression contract while
cutting decoder latency. The **encoder is v4's `MMEncoder`, unchanged** (49,928
params, architecturally identical to v2's), so the latent is `C_lat = input
channels` at `1/2^depth` — verified 64.0x spatial at depth 3 for both 1- and
3-channel inputs. Only the decoder differs.

## Measured results

120 epochs x 600 steps, fp32, default recipe, same data and held-out split as
the v2/v4 references. PSNR is the mean over 5 modalities (depth, occupancy,
costmap, heatmap, rgb) on held-out frames. Decoder latency is GPU-resident p50
at 5120x3840, depth 3, FP16 + `torch.compile`.

| model | dec params | H200 ms | A100 ms | 4x | 16x | **64x** | 256x |
|---|---|---|---|---|---|---|---|
| `min_fused_floor` | 76,440 | **0.057** | 0.080 | 32.21 | 33.74 | 30.32 | 26.62 |
| `min_fused_t0` | 374,928 | **0.133** | 0.219 | 39.24 | 38.31 | **35.28** | 31.62 |
| v4 baseline *(ref)* | 379,568 | 0.226 | 0.358 | 41.89 | 39.04 | 34.47 | 31.13 |
| v2 decoder *(ref)* | 197,172 | 4.435 | 10.875 | — | — | 37.38 | — |

**`min_fused_t0` is the recommended default: 1.70x faster than v4 on H200 and
+0.81 dB at 64x** — better on both axes. **`min_fused_floor` is the latency
floor: 4.0x faster than v4**, at a real cost of ~4.2 dB at 64x.

## Why these are faster

Decoder latency here is **not** compute-bound. At 5120x3840 v4's decoder does
1.77 GMAC against a compute roofline of 0.004 ms (H200) but measures 0.226 ms —
64x off. It is bound by kernel count and memory traffic. Fitted across variants:

    latency ~= 0.082 ms (irreducible output write) + ~0.066 ms per latent-resolution conv kernel   [A100]

So reducing *kernels*, not FLOPs, is what pays. Narrowing channels buys almost
nothing (width 8 ~= width 16; width 4 is *slower*); depthwise-separable and
grouped convs are worse because they add kernels.

Both models exploit the same two facts, neither of which changes the output:
- `Conv1x1(16 -> n*4^d) -> PixelShuffle -> Sigmoid` is 3 kernels and ~118 MB of
  traffic. Broadcasting to `(B,n,Hl,S,Wl,S)` makes the reshape a free view, so
  PixelShuffle disappears and Sigmoid rides the epilogue: **one** pointwise
  kernel writing 39.3 MB.
- PixelShuffle is a permutation, so the squash commutes through it. Applying
  Sigmoid before the reshape is provably identical (max diff 6e-8) and removes a
  full-resolution pass — worth ~1.1x on its own.

The two differ only in what precedes that tail:

| model | latent-resolution trunk | kernels |
|---|---|---|
| `min_fused_floor` | none | 1 (the fused tail) |
| `min_fused_t0` | one `Conv3x3(n->16)` + ReLU | 2 |

That single 3x3 conv costs 0.076 ms on H200 and buys **+4.96 dB at 64x** — the
best marginal trade found anywhere in the search.

**Codegen caveat:** how the fused tail is *written* decides everything. As a
reduction, inductor refuses to fuse, materialises a 1.26 GB intermediate, and
runs 4.3x *slower* than v4. As unrolled broadcast FMAs it fuses into one kernel
and wins. Same algebra, 7x latency difference.

## Usage

```python
import sys; sys.path.insert(0, "fast_decoders")
from load_example import load
enc, dec = load("min_fused_t0", device="cuda")   # or "min_fused_floor"
z = enc(x, depth)                 # x: (B,1|3,H,W) in [0,1]
y = dec(z, depth, x.shape[2], x.shape[3])
```

`python fast_decoders/load_example.py <model>` verifies interface and
compression at depths 1/3/6 for both channel counts.

## Caveats

- **Validated on torch 2.5.1+cu121 only.** `fullgraph=True` compilation
  behaviour shifts between torch versions, and the reported latencies all used
  `torch.compile`. The architectures are faster in eager mode too (1.54x and
  2.10x vs v4 on H200), so a working inductor/Triton backend is not required —
  it roughly doubles the advantage where available. See ORIN_BENCHMARK.md for
  running these on Jetson, where Triton is unreliable.
- **Weights are final-epoch, not best-epoch.** The search harness validates only
  at the end of training. On the v4 reference run, final was 0.19 dB below best.
- **Latency is H200/A100, not Orin.** The deployment target is a Jetson, and
  kernel-selection behaviour does not transfer: the `slimv2` family measured
  1.8x *faster* than v2 on A100 and 2.05x *slower* on H200, reproducibly across
  repeats. Any model you deploy must be benchmarked on the real target.
- **Not uniformly better than v4.** `min_fused_t0` trails v4 by 2.65 dB at 4x
  compression; v4's stacked latent convs earn their cost at low compression.
  Check the rate you actually ship.
- Reproduce with `sbatch search/run_variant.sh search/variants/variant_<name>.py 120`;
  full search results (31 variants) are in `search/results/` and `search/README.md`.
