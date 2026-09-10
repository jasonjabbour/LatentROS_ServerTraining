"""V5: space-to-depth encoder and single-kernel fused decoder.

Same compression contract as v2 and v4 -- the latent is `C_lat = in_ch` at
`1/2^depth` -- but both halves are rebuilt around what actually costs time on
a GPU, which is kernel count and memory traffic rather than FLOPs.

Encoder. v4's whole encoder cost was its stem, `Conv2d(n, 16, 3, stride=2)` at
FULL resolution: per-kernel profiling shows 83% of its 2.17 ms in a single
`conv2d_grouped_direct_kernel`, because cuDNN treats a 1-input-channel
convolution as a grouped conv and picks a direct, non-tensor-core path. It also
writes 157 MB at H/2. `PixelUnshuffle(u)` fixes both: the stem becomes a fat
1x1 at H/u (tensor-core-shaped, no longer Cin=1) and the dominant write shrinks
to 39 MB at H/8. Latency tracks stem output bytes near-linearly. Unshuffle
alone is one linear map per disjoint u-by-u block and loses 3.4 dB; 3x3 convs
at H/u restore cross-block mixing for ~0.17 ms and buy back +3.71 dB.
Depth 1 pins u=2, because the latent must land at exactly H/2.

Decoder. v4 emitted `Conv1x1(16 -> n*4^d) -> PixelShuffle -> Sigmoid`: three
kernels and ~118 MB. Written as one einsum with the squash applied before the
final reshape (PixelShuffle is a permutation, so the squash commutes through
it) inductor emits ONE kernel that reads the latent once and writes 39 MB once.

Measured GPU-resident, H200, depth 3, 5120x3840, FP16 + torch.compile:

    stack            encoder    decoder     total     64x dB
    v4               2.170 ms   0.226 ms   2.396 ms   35.52
    v5 balanced      0.307 ms   0.133 ms   0.440 ms   35.85
    v5 fast          0.219 ms   0.133 ms   0.352 ms   35.28

Run --bench for a CUDA timing benchmark, matching v4's benchmark protocol.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTHS = [16, 32, 64, 128, 128, 128]     # per-resolution channel counts, as v4
MAX_DEPTH = len(WIDTHS)
NCHS = (1, 3)                            # grids = 1, RGB = 3

# tier -> (unshuffle factor, extra 3x3 convs at stem resolution)
TIERS = {"balanced": (8, 2), "fast": (8, 1)}
TIER_TAGS = {"balanced": "MMV5", "fast": "MMV5F"}


class DWSep(nn.Module):
    """v4's stage, unchanged: every stage below the stem was already cheap."""

    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(ic, ic, 3, stride, 1, groups=ic, bias=False)
        self.pw = nn.Conv2d(ic, oc, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class UnshuffleStem(nn.Module):
    """PixelUnshuffle(u) then a Conv2d at H/u. Replaces v4's full-res stem."""

    def __init__(self, n, oc, u, k=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.PixelUnshuffle(u),
            nn.Conv2d(n * u * u, oc, k, padding=k // 2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.body(x)


class MMEncoder(nn.Module):
    """Space-to-depth stem, then v4's shared stages and per-exit latent heads.

    umax  largest unshuffle factor; exit depth d uses u = 2**min(d, log2(umax))
          so depth 1 still lands at exactly H/2 (the compression contract).
    extra 3x3 convs at stem resolution, applied only for u >= 4 where they are
          nearly free. This is what makes a large unshuffle quality-safe.
    """

    def __init__(self, umax=8, extra=2, k=1, widths=None, max_depth=MAX_DEPTH):
        super().__init__()
        self.max_depth = max_depth
        self.widths = list(widths or WIDTHS)
        self.umax_log = {2: 1, 4: 2, 8: 3}[umax]
        self.extra_n = extra
        self.stems = nn.ModuleDict({
            f"{n}_{ul}": UnshuffleStem(n, self.widths[ul - 1], 1 << ul, k)
            for n in NCHS for ul in range(1, self.umax_log + 1)})
        self.post = nn.ModuleDict()
        if extra:
            for ul in range(2, self.umax_log + 1):
                w = self.widths[ul - 1]
                self.post[str(ul)] = nn.Sequential(*[m for _ in range(extra) for m in
                    (nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True))])
        self.stages = nn.ModuleList([
            DWSep(self.widths[i], self.widths[i + 1], stride=2)
            for i in range(max_depth - 1)])
        self.to_latent = nn.ModuleDict({str(n): nn.ModuleList([
            nn.Conv2d(self.widths[d - 1], n, 1) for d in range(1, max_depth + 1)
        ]) for n in NCHS})

    def forward(self, x, depth):
        if not 1 <= depth <= self.max_depth:
            raise ValueError(f"depth must be between 1 and {self.max_depth}")
        n = x.shape[1]
        h, w = x.shape[-2:]
        s = 1 << depth
        ph, pw = (s - h % s) % s, (s - w % s) % s
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        ul = min(depth, self.umax_log)
        x = self.stems[f"{n}_{ul}"](x)
        if self.extra_n and str(ul) in self.post:
            x = self.post[str(ul)](x)
        for i in range(ul - 1, depth - 1):
            x = self.stages[i](x)
        return self.to_latent[str(n)][depth - 1](x)   # (B, n, H/2^d, W/2^d)


class FusedExpand(nn.Module):
    """Conv1x1(cin -> n*s^2) + PixelShuffle(s) as ONE einsum kernel.

    Mathematically identical to the conv+shuffle pair (verified bit-exact in
    fp32) but it never materialises the n*s^2-channel latent-resolution tensor,
    which is the intermediate v4's PixelShuffle has to write and read back.
    """

    def __init__(self, cin, n, s):
        super().__init__()
        self.n, self.s = n, s
        self.weight = nn.Parameter(torch.empty(n, cin, s, s))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(n, s, s))

    def forward(self, x):
        y = torch.einsum("nihw,oiac->nohawc", x, self.weight)
        return y + self.bias.view(1, self.n, 1, self.s, 1, self.s)


class MMDecoder(nn.Module):
    """One Conv3x3(n->16)+ReLU at latent resolution, then a single fused
    expansion. The Sigmoid is applied BEFORE the final reshape: PixelShuffle is
    a permutation, so the squash commutes through it and this removes a
    full-resolution pass."""

    latent_halo = 3

    def __init__(self, stem=16, max_depth=MAX_DEPTH):
        super().__init__()
        if not 1 <= max_depth <= MAX_DEPTH:
            raise ValueError(f"max_depth must be between 1 and {MAX_DEPTH}")
        self.max_depth = max_depth
        self.from_latent = nn.ModuleDict({str(n): nn.ModuleList([
            nn.Sequential(nn.Conv2d(n, stem, 3, padding=1), nn.ReLU(inplace=True))
            for _ in range(max_depth)]) for n in NCHS})
        self.expand = nn.ModuleDict({str(n): nn.ModuleList([
            FusedExpand(stem, n, 2 ** d) for d in range(1, max_depth + 1)])
            for n in NCHS})

    def forward(self, z, depth, th=None, tw=None):
        if not 1 <= depth <= self.max_depth:
            raise ValueError(f"depth must be between 1 and {self.max_depth}")
        if z.ndim != 4 or z.shape[1] not in NCHS:
            raise ValueError("latent must be NCHW with 1 or 3 channels")
        if (th is None) != (tw is None):
            raise ValueError("th and tw must be specified together")
        n = z.shape[1]
        x = self.from_latent[str(n)][depth - 1](z)
        y = torch.sigmoid(self.expand[str(n)][depth - 1](x))
        s = 2 ** depth
        y = y.reshape(y.shape[0], n, x.shape[2] * s, x.shape[3] * s)
        if th is not None:
            if not (0 < th <= y.shape[2] and 0 < tw <= y.shape[3]):
                raise ValueError("crop must fit within the reconstructed grid")
            y = y[:, :, :th, :tw]
        return y


def build(tier="balanced"):
    """The matched pair for a tier. Encoder architecture differs per tier, so
    weights are NOT interchangeable between tiers."""
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; expected one of {tuple(TIERS)}")
    umax, extra = TIERS[tier]
    return MMEncoder(umax=umax, extra=extra), MMDecoder()


def benchmark():
    """Time CPU-array round trips and GPU-resident whole-frame decoding."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from time import perf_counter

    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--size', choices=('small', 'large', 'both'), default='both')
    parser.add_argument('--depth', type=int, choices=range(1, MAX_DEPTH + 1), default=3)
    parser.add_argument('--weights', type=Path, default=Path(__file__).parent / 'weights')
    parser.add_argument('--tier', choices=tuple(TIERS), default='balanced',
                        help='balanced (u=8, 2 extra convs) or fast (u=8, 1 extra conv)')
    parser.add_argument('--tag', default=None,
                        help='Checkpoint filename suffix; defaults to the tier tag (MMV5 / MMV5F)')
    parser.add_argument('--random-weights', action='store_true', help='Time untrained models without checkpoints')
    parser.add_argument('--input', type=Path, help='5120x3840 int8 NPY or NPZ with a costmap key; otherwise synthetic')
    parser.add_argument('--output', type=Path, help='Append to one results JSON; otherwise print only')
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if not args.bench or min(args.warmups, args.repeats) < 1:
        parser.error('--bench and positive warmups/repeats are required')
    if args.tag is None:
        args.tag = TIER_TAGS[args.tier]
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU fallback')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.config.suppress_errors = False
    if args.input:
        saved = np.load(args.input, allow_pickle=False)
        if isinstance(saved, np.lib.npyio.NpzFile):
            with saved as capture:
                saved = capture['costmap']
    else:
        saved = np.random.default_rng(42).integers(0, 101, (5120, 3840), dtype=np.int8)
    if saved.shape != (5120, 3840) or saved.dtype != np.int8:
        parser.error('--input must contain a 5120x3840 int8 grid')
    if saved.min() < 0 or saved.max() > 100:
        parser.error('--input values must be between 0 and 100')
    encoder, decoder = build(args.tier)
    weight_files = [] if args.random_weights else [
        args.weights / f'{kind}_best_{args.tag}.pth' for kind in ('encoder', 'decoder')]
    for model, path in zip((encoder, decoder), weight_files):
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    encoder, decoder = encoder.cuda().half().eval(), decoder.cuda().half().eval()
    # Fixed-shape whole-frame calls; compilation is excluded from timed samples.
    encode_gpu = torch.compile(lambda x: encoder(x, args.depth), dynamic=False, fullgraph=True)
    decode_gpu = torch.compile(lambda z: decoder(z, args.depth), dynamic=False, fullgraph=True)
    sources = [Path(__file__), *weight_files]
    run = {'model': 'v5', 'tier': args.tier, 'random_weights': args.random_weights, 'seed': 42,
           'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'cudnn': torch.backends.cudnn.version(), 'cpu_threads': torch.get_num_threads(),
           'compile': True, 'dynamic': False, 'fp16': True, 'strip_rows': 0, 'depth': args.depth,
           'warmups': args.warmups, 'repeats': args.repeats,
           'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
           'cases': [], 'completed': False}
    report = json.loads(args.output.read_text()) if args.output and args.output.exists() else {'runs': []}
    report['runs'].append(run)

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')

    print(f"{run['gpu']}: v5 [{args.tier}], FP16 torch.compile, depth {args.depth}, whole-frame; "
          f"random weights={args.random_weights}. Compilation/warmup excluded.", flush=True)
    try:
        with torch.no_grad():
            for stride in ((2, 1) if args.size == 'both' else (2 if args.size == 'small' else 1,)):
                data = torch.from_numpy(np.ascontiguousarray(saved[::stride, ::stride]))[None, None]
                h, w = data.shape[-2:]
                scale = 2 ** args.depth

                def encode():
                    x = data.cuda().float().div_(100).half()
                    return encode_gpu(x).cpu().numpy().tobytes()

                def unpack(payload):
                    return torch.frombuffer(bytearray(payload), dtype=torch.float16).reshape(
                        1, 1, (h + scale - 1)//scale, (w + scale - 1)//scale).cuda()

                def decode(payload):
                    x = decode_gpu(unpack(payload))[..., :h, :w]
                    return x.float().mul_(100).round_().clamp_(0, 100).to(torch.int8).cpu()

                case = {'shape': [h, w], 'native_bytes': data.numel(),
                        'input_sha256': hashlib.sha256(data.numpy().tobytes()).hexdigest(), 'samples': []}
                run['cases'].append(case)
                print(f'{h}x{w}: warming {args.warmups} calls, then measuring {args.repeats}...', flush=True)
                for i in range(args.warmups + args.repeats):
                    if i == args.warmups:
                        torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = perf_counter()
                    payload = encode()
                    torch.cuda.synchronize()
                    encoded = perf_counter()
                    output = decode(payload)
                    torch.cuda.synchronize()
                    done = perf_counter()
                    if i >= args.warmups:
                        case['samples'].append({'encode_ms': (encoded-start)*1000,
                                                'decode_ms': (done-encoded)*1000, 'total_ms': (done-start)*1000})
                case['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                case['latent_bytes'] = len(payload)
                assert output.shape == data.shape and output.dtype == torch.int8
                assert len(payload) == 2*((h + scale - 1)//scale)*((w + scale - 1)//scale)
                assert np.isfinite(np.frombuffer(payload, np.float16)).all()
                latent = unpack(payload)
                assert torch.isfinite(decode_gpu(latent)).all().item()
                gpu_x = data.cuda().float().div_(100).half()
                for sample in case['samples']:
                    torch.cuda.synchronize()
                    start = perf_counter()
                    decode_gpu(latent)
                    torch.cuda.synchronize()
                    sample['decoder_gpu_resident_ms'] = (perf_counter()-start)*1000
                    torch.cuda.synchronize()
                    start = perf_counter()
                    encode_gpu(gpu_x)
                    torch.cuda.synchronize()
                    sample['encoder_gpu_resident_ms'] = (perf_counter()-start)*1000
                del gpu_x
                for stat, percentile in (('p50_ms', 50), ('p95_ms', 95)):
                    case[stat] = {k: float(np.percentile([s[k] for s in case['samples']], percentile))
                                  for k in case['samples'][0]}
                print(f"{h}x{w}: p50={case['p50_ms']}; p95={case['p95_ms']}; "
                      f"peak allocated={case['peak_allocated_bytes']/1e6:.1f} MB", flush=True)
                save()
                del latent, data, output
        run['completed'] = True
    except Exception as error:
        run['error'] = str(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    benchmark()
