# Unified model v5 — space-to-depth encoder, single-kernel decoder

Same compression contract as v2 and v4: one autoencoder, any 2D grid message,
selectable rate `4^d` at exit depth `d`, latent `C_lat = in_ch` at `1/2^depth`.
Both halves are rebuilt around what actually costs GPU time — kernel count and
memory traffic, not FLOPs.

Two tiers, selected with `--tier`:

| tier | encoder | tag | H200 enc ms | H200 dec ms | total |
|---|---|---|---|---|---|
| `balanced` | unshuffle 8 + two 3x3 at H/8 | `MMV5` | **0.307** | 0.133 | **0.440 ms** |
| `fast` | unshuffle 8 + one 3x3 at H/8 | `MMV5F` | **0.219** | 0.133 | **0.352 ms** |
| *v4 (reference)* | | `MMV4` | 2.170 | 0.226 | 2.396 ms |

## Results (held-out PSNR, dB — mean over 5 modalities, 120 epochs x 600 steps)

| rate | v4 | v5 balanced | v5 fast |
|---|---|---|---|
| 4x | 39.93 | **41.98** | 40.71 |
| 16x | 38.43 | **40.22** | 39.05 |
| 64x | 35.52 | **35.85** | 35.67 |
| 256x | 31.60 | **32.26** | 31.46 |

`balanced` is **5.4x faster than v4 end-to-end and higher at all four rates**.
`fast` is 6.8x faster and higher at three of four.

Per-modality at 64x (`balanced`): heatmap 46.77, costmap 42.82, occupancy 34.23,
depth 30.10, rgb 25.34.

Cross-GPU: `balanced` reads **0.659 ms on A100 against a 34.83 ms v4 baseline**.
The A100 encoder baseline is 16x the H200 number — not a scaling factor — because
a different cuDNN kernel dominates there. Rankings invert between the two GPUs,
so **do not screen encoder work on A100**.

## Architecture

- **Encoder** — `PixelUnshuffle(u)` then a fat `1x1` at `H/u`, then `extra` 3x3
  convs at that resolution, then v4's shared depthwise-separable stride-2 stages
  (widths 16/32/64/128/128/128) and a per-exit `1x1` latent head. Exit depth `d`
  uses `u = 2**min(d, 3)`, so depth 1 still lands at exactly `H/2`.
- **Decoder** — `Conv3x3(n->16)`+ReLU at latent resolution, then one fused
  expansion: `Conv1x1(16 -> n*4^d) + PixelShuffle(2^d)` written as a single
  einsum, with the Sigmoid applied *before* the final reshape.
- **Interface** — `MMEncoder(x, depth)`, `MMDecoder(z, depth, H, W)`; NCHW with
  1 or 3 channels, reflect-padded to a multiple of `2^depth` and cropped back.
- `build(tier)` returns the matched pair. **Weights are not interchangeable
  between tiers** (the encoders differ) or with v4.

### Why v4's encoder was slow

Its whole cost was the stem, `Conv2d(n, 16, 3, stride=2)` at full resolution:
cost was flat across exit depths (2.329 / 2.172 / 2.255 ms at depths 1/3/6)
because every later stage runs at a quarter the spatial extent. Per-kernel
profiling shows **83% of the 2.17 ms in one `conv2d_grouped_direct_kernel`** —
cuDNN treats a 1-input-channel convolution as a *grouped* conv and picks a
direct, non-tensor-core path. It also writes 157 MB at H/2.

`PixelUnshuffle` fixes both: the stem conv becomes tensor-core-shaped (no longer
`Cin=1`; unshuffle-2 alone recovers 3.5x at the *same* write), and the dominant
write drops to 39 MB at H/8. Latency tracks stem output bytes near-linearly
(157 MB -> 0.63 ms, 79 -> 0.25, 39 -> 0.13). Unshuffle alone is one linear map
per disjoint 8x8 block and loses 3.4 dB; the 3x3 convs at H/8 restore
cross-block mixing for ~0.17 ms and buy back **+3.71 dB**.

## Training

```bash
python3.10 -m venv ~/venvs/codec-v5 && source ~/venvs/codec-v5/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==2.2.6 Pillow==9.4.0

# one-time: pre-decode RGB PNGs to a uint8 npy cache (lossless, ~11 s)
python data_multimodal.py --prep-rgb /data/rgb --rgb-dir /data/rgb_npy \
  --depth-dir /data/depth --derived-dir /data/derived_occ --workers 28

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_multimodal.py --tier balanced \
  --depth-dir /data/depth --rgb-dir /data/rgb_npy --derived-dir /data/derived_occ \
  --workers 28 --prefetch 8 --output runs/v5
```

Recipe is v4's: 120 epochs x 600 steps, MSE, Adam 1e-3, cosine decay, FP32,
five modalities, six exits, random sizes, the same 90/10 split. Three mechanical
changes make it ~34x faster wall-clock at equal quality: RGB from the npy cache
(43 ms/sample of PNG decode removed), cast/normalise/resize on the GPU, and
batches built by worker **processes** rather than a GIL-bound thread pool. Stage
the dataset on node-local disk — NFS reads were the original bottleneck.

`python data_multimodal.py --check ...` builds one batch per modality and prints
shapes and value ranges.

## Benchmarking

```bash
python multimodal_ae.py --bench --tier balanced --size both --depth 3 \
  --output timing.json
```

FP16 + `torch.compile`, whole-frame, compilation and warmup excluded. Reports
p50/p95 encode/decode/total including host transfers, plus **both**
`decoder_gpu_resident_ms` and `encoder_gpu_resident_ms` — v4's benchmark
isolated only the decoder, which hid the fact that the encoder was 94% of codec
GPU compute. `--tag` defaults to the tier's tag; `--random-weights` times an
untrained model.

## Caveats

- **Quality margins are single-seed for `balanced`.** Training seeds data order
  but **not** model init. Three seeds of the `fast` encoder gave 34.76 / 35.28 /
  35.67 dB at 64x — a 0.91 dB spread — so differences below ~1 dB between
  architectures are not resolvable from one run each. The latency figures are
  reproducible to ~1% and need no such hedging.
- **Weights here are final-epoch** (the architecture search harness validated
  only at the end); `train_multimodal.py` in this directory does save
  best-mean-PSNR checkpoints, so a fresh run gives a true best-epoch model.
- **Latency is H200/A100, not Orin.** See `fast_decoders/ORIN_BENCHMARK.md`.
  Kernel selection does not transfer, in either magnitude or direction.
- **Quality is measured at 256x256 / 384x640 / 720x1280; latency at 5120x3840.**
  Different operating points; PSNR at the benchmark size is unverified.
- Validated on torch 2.5.1+cu121. The speedups hold in eager mode too (the
  decoder is 1.54x vs v4 without `torch.compile`), so a Triton backend is not
  required, which matters on aarch64.

## Provenance

Derived from a 118-variant architecture search (73 encoder, 45 decoder) in
`search/`; see `search/README.md`. Negative results recorded there so they are
not retried: `channels_last` did nothing; a 1x1 stride-2 stem is not faster;
reduction and ConvTranspose formulations lose to algebraically identical
alternatives because inductor materialises instead of fusing; widening
low-resolution stages is not free (+0.35 ms at width 256); half resolution is
not free (+1.6 ms for two 16->16 3x3 convs at H/2); and the decoder's fused-tail
trick *inverts* on the encoder side — 1.5x better at unshuffle 2, 10x worse at
unshuffle 8, because 64 unrolled terms stop fusing.
