"""V4: six-exit encoder and latent-resolution decoder.

Each exit has input/output heads around two shared 16-channel convolutions.
The final PixelShuffle directly reconstructs the original resolution, avoiding
wide full-resolution feature maps. Run --bench for a CUDA timing benchmark.
"""
import torch.nn as nn
import torch.nn.functional as F

WIDTHS = [16, 32, 64, 128, 128, 128]
MAX_DEPTH = len(WIDTHS)
NCHS = (1, 3)


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
        self.stem = nn.ModuleDict({str(n): nn.Sequential(
            nn.Conv2d(n, WIDTHS[0], 3, 2, 1), nn.ReLU(inplace=True)) for n in NCHS})
        self.stages = nn.ModuleList([
            DWSep(WIDTHS[d - 2], WIDTHS[d - 1], stride=2)
            for d in range(2, max_depth + 1)])
        self.to_latent = nn.ModuleDict({str(n): nn.ModuleList([
            nn.Conv2d(WIDTHS[d - 1], n, 1) for d in range(1, max_depth + 1)
        ]) for n in NCHS})

    def forward(self, x, depth):
        n = x.shape[1]
        h, w = x.shape[-2:]
        s = 1 << depth
        ph, pw = (s - h % s) % s, (s - w % s) % s
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        x = self.stem[str(n)](x)
        for i in range(depth - 1):
            x = self.stages[i](x)
        return self.to_latent[str(n)][depth - 1](x)


class MMDecoder(nn.Module):
    latent_halo = 3  # Three latent-resolution 3x3 convolutions before PixelShuffle.

    def __init__(self, max_depth=MAX_DEPTH):
        super().__init__()
        if not 1 <= max_depth <= MAX_DEPTH:
            raise ValueError(f"max_depth must be between 1 and {MAX_DEPTH}")
        self.max_depth = max_depth
        self.from_latent = nn.ModuleDict({str(n): nn.ModuleList([
            nn.Sequential(nn.Conv2d(n, 16, 3, padding=1), nn.ReLU(inplace=True))
            for _ in range(max_depth)
        ]) for n in NCHS})
        self.shared = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.to_pixels = nn.ModuleDict({str(n): nn.ModuleList([
            nn.Sequential(nn.Conv2d(16, n * 4 ** d, 1),
                          nn.PixelShuffle(2 ** d), nn.Sigmoid())
            for d in range(1, max_depth + 1)
        ]) for n in NCHS})

    def forward(self, z, depth, th=None, tw=None):
        if not 1 <= depth <= self.max_depth:
            raise ValueError(f"depth must be between 1 and {self.max_depth}")
        if z.ndim != 4 or z.shape[1] not in NCHS:
            raise ValueError("latent must be NCHW with 1 or 3 channels")
        if (th is None) != (tw is None):
            raise ValueError("th and tw must be specified together")
        n = str(z.shape[1])
        x = self.from_latent[n][depth - 1](z)
        x = self.to_pixels[n][depth - 1](self.shared(x))
        if th is not None:
            if not (0 < th <= x.shape[2] and 0 < tw <= x.shape[3]):
                raise ValueError("crop must fit within the reconstructed grid")
            x = x[:, :, :th, :tw]
        return x


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
    parser.add_argument('--tag', default='MMV4', help='Checkpoint filename suffix')
    parser.add_argument('--random-weights', action='store_true', help='Time untrained models without checkpoints')
    parser.add_argument('--input', type=Path, help='5120x3840 int8 NPY or NPZ with a costmap key; otherwise synthetic')
    parser.add_argument('--output', type=Path, help='Append to one results JSON; otherwise print only')
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if not args.bench or min(args.warmups, args.repeats) < 1:
        parser.error('--bench and positive warmups/repeats are required')
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
    encoder, decoder = MMEncoder(), MMDecoder()
    weight_files = [] if args.random_weights else [
        args.weights / f'{kind}_best_{args.tag}.pth' for kind in ('encoder', 'decoder')]
    for model, path in zip((encoder, decoder), weight_files):
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    encoder, decoder = encoder.cuda().half().eval(), decoder.cuda().half().eval()
    # Fixed-shape whole-frame calls; compilation is excluded from timed samples.
    encode_gpu = torch.compile(lambda x: encoder(x, args.depth), dynamic=False, fullgraph=True)
    decode_gpu = torch.compile(lambda z: decoder(z, args.depth), dynamic=False, fullgraph=True)
    sources = [Path(__file__), *weight_files]
    run = {'model': 'v4', 'random_weights': args.random_weights, 'seed': 42,
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

    print(f"{run['gpu']}: v4, FP16 torch.compile, depth {args.depth}, whole-frame; "
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
                for sample in case['samples']:
                    torch.cuda.synchronize()
                    start = perf_counter()
                    decode_gpu(latent)
                    torch.cuda.synchronize()
                    sample['decoder_gpu_resident_ms'] = (perf_counter()-start)*1000
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
