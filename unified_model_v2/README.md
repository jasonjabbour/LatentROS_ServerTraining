# Unified codec v2 — variable-rate, any-size, multi-modality

One learned autoencoder that compresses **any 2D-grid robot message** to a compact latent and reconstructs
it, at a **selectable rate**, across **modalities** (depth, BEV occupancy, nav cost-map, heat-map, RGB) and
**arbitrary input sizes**. Rate is the spatial **exit depth** `d` (ratio `4^d`). Successor to `unified_model/`
(depth-only, single-size); kept separate until it supersedes it.

## Architecture
Early-exit shared trunk (as v1) with multi-modality I/O:
- **Encoder** — a per-channel **stem** (`1→16` for grids, `3→16` for RGB) does the first downsample, then a
  **shared** depthwise-separable stride-2 stack (widths 16,32,64,128,128,128) with a latent head after each
  stage. Encode at rate `4^d` = stem + first `d−1` shared stages + head `d`.
- **Latent** — **matched to the input channels** (`C_lat = in_ch`: 1 for grids, 3 for RGB), so every modality
  gets the same ratio `4^d` at a given depth, and RGB keeps enough capacity for colour.
- **Decoder** — mirror: per-channel entry, `d` **shared** upsample stages (`DWSep → PixelShuffle(2)`) whose
  width stays wide at full resolution (`DEC_WIDTHS` ends at 32) for fidelity, then a per-channel `Sigmoid` head.
- Only the thin stems/heads are per-modality; the whole learned core is shared. Fully convolutional → any `H×W`
  (reflect-pad to a multiple of `2^d`). ~247K parameters total.

## Training recipe (appendix)
| | |
|---|---|
| Hardware | 1× NVIDIA RTX 3080 Ti (12 GB), CUDA; PyTorch 2.5.1, conda env `latentros` |
| Data | **depth** 5000 CARLA (720×1280); **RGB** 4542 collected in CARLA at 640×480 / 1280×720 / 1920×1080; **occupancy / cost-map / heat-map** derived from the depth frames (back-project → BEV occupancy → distance inflation → smoothing), occupancy cached once |
| Sampling (per step) | draw a **modality** (weights depth/RGB ×2 vs the derived grids), a random **size** (short side 128–768, aspect ∈ {1:1, 4:3, 16:9} + portraits — *multi-scale*), and an exit-depth **rate** (low-rate-weighted). Batch size scales inversely with area (fixed ~1.6M-pixel budget) so memory is constant across sizes |
| Normalization | depth clip 50 m ÷50; occupancy/cost/heat already [0,1]; RGB ÷255 |
| Objective | pixel MSE |
| Optimizer | Adam, lr 1e-3, cosine decay to 0 |
| Schedule | 120 epochs × 600 steps, background-prefetched loader; ~48 min wall-clock (best at ep 104) |
| Size generalization | evaluated at held-out sizes incl. **1280×720 / 1920×1080 (larger than any training size)** to test extrapolation |

## Results (held-out PSNR, dB — one model)
Reconstruction is strong at usable rates across every modality:

| rate | depth | occupancy | costmap | heatmap | rgb |
|---|---|---|---|---|---|
| 4× | 38.2 | 45.6 | 51.5 | 50.2 | 34.5 |
| 16× | 35.1 | 41.5 | 51.6 | 53.3 | 32.1 |
| 64× | 32.0 | 38.5 | 47.3 | 51.5 | 28.1 |
| 256× | 29.3 | 30.8 | 41.5 | 47.7 | 24.3 |

**Size generalization** (16×, depth/rgb; `*` = larger than any training size — the model *extrapolates* and
even improves with more spatial context):

| size | 256² | 768×512 | 1280×720* | 1920×1080* |
|---|---|---|---|---|
| depth | 30.5 | 35.1 | 35.3 | 40.0 |
| rgb | 27.5 | 32.1 | 33.7 | 34.5 |

## Files
| file | what |
|---|---|
| `multimodal_ae.py` | the model (`MMEncoder`/`MMDecoder`); `--bench` prints per-modality/size/rate latency |
| `data_multimodal.py` | modality loaders + depth→grid derivation; `--prep` builds the occupancy cache, `--check` sanity-loads |
| `train_multimodal.py` | the multi-modal / multi-scale / multi-rate trainer |
| `weights/` | checkpoints `{encoder,decoder}_best_MMV2.pth` |
| `derived_occ/` | cached BEV occupancy (one per depth frame) |
| `reconstruct_playground.ipynb` | reconstruct any modality at any size/rate; galleries + rate-distortion |

## Reproduce
```bash
conda activate latentros
python data_multimodal.py --prep                                             # build occupancy cache (once)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_multimodal.py --epochs 120 --steps 600
python multimodal_ae.py --bench                                              # latency
# then open reconstruct_playground.ipynb
```
