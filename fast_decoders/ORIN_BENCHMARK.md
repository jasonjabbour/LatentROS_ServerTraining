# Benchmarking these decoders on Jetson Orin — instructions for an agent

You are measuring codec latency on an NVIDIA Jetson Orin. All numbers below
were measured on datacenter GPUs (H200/A100) and **do not transfer** — the whole
point of this task is to get real Orin numbers.

## 1. Get the code

```bash
git clone git@github.com:jasonjabbour/LatentROS_ServerTraining.git
cd LatentROS_ServerTraining          # main branch has fast_decoders/
```

## 2. Environment — do NOT pip install torch

`pip install torch==2.5.1 --index-url .../cu121` is x86-only and will fail or
install a CPU build. On Jetson use NVIDIA's aarch64 wheel matching the installed
JetPack, or the `nvcr.io/nvidia/l4t-pytorch` container. Verify before proceeding:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```
`cuda.is_available()` must be True. Also `pip install numpy pillow`.

## 3. Lock the clocks — otherwise your numbers are noise

Orin defaults to a power-limited, dynamically-clocked state. Without this step
latency varies by 2x+ between runs and is not comparable to anything.

```bash
sudo nvpmodel -m 0        # max-power mode (MAXN)
sudo jetson_clocks        # pin CPU/GPU/EMC clocks to max
sudo nvpmodel -q          # confirm the mode took effect
```
Run `tegrastats` in a second terminal during benchmarking and **report whether
GPU frequency dropped or any thermal throttling appeared** — if it did, the
numbers are invalid and you should let the device cool and rerun.

## 4. Run the decoder benchmark

Latency is weight-independent, so no training or dataset is needed.

```bash
python search/bench_only.py \
  --variant search/variants/variant_baseline_v4.py \
  --variant fast_decoders/min_fused_t0/architecture.py \
  --variant fast_decoders/min_fused_floor/architecture.py \
  --sizes 5120x3840,2560x1920
```

**Always include `variant_baseline_v4.py` in the same invocation.** The script
prints speedups against a hardcoded x86 reference, which is meaningless on Orin
— ignore those labels and compute ratios against the baseline_v4 number
measured in your own run.

If it aborts after a few variants with a `torch._dynamo` cache error, run one
variant per process instead.

**Then repeat with `--no-compile`.** `torch.compile` needs a working Triton
backend, which is unreliable on aarch64. If compilation fails or is slow to
warm up, the eager numbers are the deployable ones. On H200 the fast decoders
win in eager too (1.54x and 2.10x vs v4), so report both.

## 5. Run the encoder benchmark

The encoder is ~94% of total GPU compute on H200 (2.168 ms vs the decoder's
0.132 ms) and has never been optimized, so this is the more important number:

```bash
python search/bench_enc.py --variant search/enc_variants/enc_baseline_v4.py
```

## 6. End-to-end, including host transfer

```bash
python unified_model_v4/multimodal_ae.py --bench \
  --weights fast_decoders/min_fused_t0 --tag final_min_fused_t0 \
  --size both --depth 3 --output orin_timing.json
```
If the `--tag` doesn't match the checkpoint filenames, rename the weights to
`encoder_best_MMV4.pth` / `decoder_best_MMV4.pth` in a scratch directory and
point `--weights` at it. This path is host-transfer bound and Orin's unified
memory may behave very differently from a discrete GPU — that difference is one
of the most interesting things you can report.

## 7. What to report

A table of p50/p95 for each model at both sizes, with:
- **compiled vs eager** side by side
- ratios computed against **your own** baseline_v4 measurement, not the x86 numbers
- the encoder number from step 5
- whether clocks stayed pinned (from `tegrastats`)
- JetPack version, torch version, and Orin model/memory

Compare against the existing Orin rows in `unified_model_v4/benchmark.json`
(depth 3: 14.66 ms total at 2560x1920, 55.00 ms at 5120x3840) — those are from a
previous generation of this codec and are the only prior on-device data.

## Reference numbers to beat (H200, depth 3, 5120x3840, GPU-resident, compiled)

| component | ms |
|---|---|
| v4 encoder | 2.168 |
| v4 decoder | 0.226 |
| min_fused_t0 decoder | 0.133 |
| min_fused_floor decoder | 0.057 |

Do not assume these ratios hold. On this project a decoder family measured 1.8x
*faster* than a reference on A100 and 2.05x *slower* on H200, reproducibly —
kernel selection does not transfer across hardware, and Orin is further away
than either.
