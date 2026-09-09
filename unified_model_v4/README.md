# Unified model v4

## Training

Copy this directory only. Use Python 3.10 and a CUDA 12.1-compatible NVIDIA
GPU driver. From inside the copied directory:

```bash
python3.10 -m venv ~/venvs/codec-v4
source ~/venvs/codec-v4/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==2.2.6 Pillow==9.4.0

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_multimodal.py \
  --depth-dir /data/depth --rgb-dir /data/rgb --derived-dir /data/derived_occ \
  --workers 16 --prefetch 8 --pin-memory --torch-threads 1 --output runs/v4
```

Replace the three dataset paths with your locations:

| Argument | Files | Used for |
|---|---|---|
| `--depth-dir` | 5,000 depth `.npy` frames | Depth training |
| `--rgb-dir` | 4,542 RGB `.png` frames | RGB training |
| `--derived-dir` | 5,000 cached occupancy `.npy` maps | Occupancy directly; costmaps by inflation; heatmaps by smoothing |

Defaults match the original recipe: **120 epochs × 600 steps**, both networks
trained from scratch, MSE, Adam at 0.001, cosine decay, FP32, five modalities,
six exits, random sizes and the same 90/10 split. Batch size is
`max(2, min(16, 1600000 // (H*W)))`.

On one A100, the command uses 16 loading threads and prefetched, pinned batches.
Optional `--precision bf16` changes training precision; validation stays FP32.
Optional `--budget 6400000 --batch-cap 64` increases batch size and changes the
recipe. Set workers to fit your CPU allocation.

Each fresh `--output` directory contains:

- `encoder_best_MMV4.pth` and `decoder_best_MMV4.pth`: best mean-validation-PSNR weights.
- `history.json`: configuration, per-epoch training MSE, validation MSE/PSNR
  by modality, sampled modality counts, learning rate, time and best epoch.
  Updated every epoch, so finished epochs remain available after interruption.

## Benchmarking

After training, benchmark the saved weights at both map sizes:

```bash
python multimodal_ae.py --bench --weights runs/v4 --size both --depth 3
```

Uses **FP16 + torch.compile**, 4.9 MB and 19.66 MB synthetic int8 maps, and
whole-frame decoding. Prints encode/decode/total p50/p95 including host transfers,
plus GPU-resident decoder time. Compilation and warmup are excluded. Use
`--input /data/map.npy` for a 5120×3840 int8 map, or `.npz` with a `costmap` key.
Use `--output timing.json` to save timings; `--random-weights` explicitly tests
an untrained model. This standalone benchmark does not measure network latency.

## Model

- **Encoder:** stride-2 stem and depthwise-separable stages with widths
  `16/32/64/128/128/128`; a latent head at each of six exits.
- **Decoder:** latent-resolution `3×3 Conv(C,16)`, two shared `3×3 Conv(16,16)`
  layers, then `1×1 Conv(16,C×4^d) → PixelShuffle(2^d) → Sigmoid`. ReLU follows
  each 3×3 convolution; input/output heads are per channel count and exit.
- **Interface:** `MMEncoder(x, depth)` and `MMDecoder(z, depth, H, W)`;
  NCHW input with one or three channels, padding/cropping to preserve dimensions.
- At exit `d`, the latent contains about `4^d` fewer elements. Depth 3 gives
  **32× fewer bytes** for an int8 map encoded as FP16: 153,600 / 614,400 bytes
  for the two benchmark sizes. Reconstruction quality requires training.
