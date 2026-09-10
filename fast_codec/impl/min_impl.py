"""Decoder implementations for the variant_min_* ladder (latency-floor study).

Lives in an importable sibling module on purpose: a class defined inside a
path-loaded variant_*.py has __module__ == "variant_<stem>", which dynamo has
to re-import when a traced forward() hits its first LOAD_GLOBAL.  Classes here
have __module__ == "min_impl", which is importable from search/variants/.

One parametric class covers the whole ladder so every rung differs only in
config, never in code path:

    stem     None -> the expansion head reads the latent directly (no trunk at
             all); int C -> Conv3x3(n->C)+ReLU at latent resolution.
    trunk    per-conv output widths of the *shared* latent-resolution stack
             (v4 == (16, 16) on top of stem=16).
    dw       shared trunk convs become depthwise-separable (3x3 dw + 1x1 pw).
    groups   grouped 3x3 trunk convs instead (1 == dense).
    expand   'ps' -> Conv1x1(C -> n*4^d) + PixelShuffle(2^d)   (v4's scheme)
             'ct' -> ConvTranspose2d(C -> n, 2^d, stride=2^d), which is the
             same linear map done in a single kernel: it never materialises the
             n*4^d-channel latent-resolution tensor, so it reads ~0.6 MB and
             writes the output once instead of writing+reading ~39 MB.
    presig   True  -> the squashing nonlinearity is applied BEFORE PixelShuffle.
             PixelShuffle is a pure permutation, so sigmoid(PS(x)) == PS(sigmoid(x))
             exactly; doing it first lets inductor fuse it and removes one
             full-resolution read+write pair (~78 MB at 5120x3840).
             Ignored for expand='ct' (nothing to commute through).
    tail     'sigmoid' | 'clamp' | 'hardsigmoid'.
"""
import torch
import torch.nn as nn

NCHS = (1, 3)
MAX_DEPTH = 6


class _DWSep(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.dw = nn.Conv2d(ic, ic, 3, 1, 1, groups=ic, bias=False)
        self.pw = nn.Conv2d(ic, oc, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class _Squash(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        if self.kind == "sigmoid":
            return torch.sigmoid(x)
        if self.kind == "hardsigmoid":
            return torch.clamp(x * 0.16666667 + 0.5, 0.0, 1.0)
        return torch.clamp(x, 0.0, 1.0)


class MinDecoder(nn.Module):
    """Configurable minimal latent-resolution decoder; see module docstring."""

    latent_halo = 3

    def __init__(self, stem=16, trunk=(16, 16), dw=False, groups=1,
                 expand="ps", presig=False, tail="sigmoid", max_depth=MAX_DEPTH):
        super().__init__()
        assert expand in ("ps", "ct")
        assert tail in ("sigmoid", "clamp", "hardsigmoid")
        self.max_depth = max_depth
        self.expand = expand
        self.presig = presig and expand == "ps"

        if stem is None:
            assert not trunk, "a trunk needs a stem to feed it"
            self.from_latent = None
            widths = {n: n for n in NCHS}
        else:
            self.from_latent = nn.ModuleDict({str(n): nn.ModuleList([
                nn.Sequential(nn.Conv2d(n, stem, 3, padding=1), nn.ReLU(inplace=True))
                for _ in range(max_depth)]) for n in NCHS})
            widths = {n: stem for n in NCHS}

        layers, c = [], stem
        for oc in trunk:
            if dw:
                layers.append(_DWSep(c, oc))
            else:
                g = groups if (c % groups == 0 and oc % groups == 0) else 1
                layers += [nn.Conv2d(c, oc, 3, padding=1, groups=g), nn.ReLU(inplace=True)]
            c = oc
        self.shared = nn.Sequential(*layers) if layers else None
        if trunk:
            widths = {n: c for n in NCHS}

        squash = _Squash(tail)
        if expand == "ps":
            def head(n, d):
                cin = widths[n]
                if self.presig:
                    return nn.Sequential(nn.Conv2d(cin, n * 4 ** d, 1), _Squash(tail),
                                         nn.PixelShuffle(2 ** d))
                return nn.Sequential(nn.Conv2d(cin, n * 4 ** d, 1),
                                     nn.PixelShuffle(2 ** d), _Squash(tail))
        else:
            def head(n, d):
                s = 2 ** d
                return nn.Sequential(nn.ConvTranspose2d(widths[n], n, s, stride=s), _Squash(tail))
        self.to_pixels = nn.ModuleDict({str(n): nn.ModuleList([
            head(n, d) for d in range(1, max_depth + 1)]) for n in NCHS})
        del squash

    def forward(self, z, depth, th=None, tw=None):
        if not 1 <= depth <= self.max_depth:
            raise ValueError(f"depth must be between 1 and {self.max_depth}")
        if z.ndim != 4 or z.shape[1] not in NCHS:
            raise ValueError("latent must be NCHW with 1 or 3 channels")
        if (th is None) != (tw is None):
            raise ValueError("th and tw must be specified together")
        n = str(z.shape[1])
        x = z
        if self.from_latent is not None:
            x = self.from_latent[n][depth - 1](x)
        if self.shared is not None:
            x = self.shared(x)
        x = self.to_pixels[n][depth - 1](x)
        if th is not None:
            if not (0 < th <= x.shape[2] and 0 < tw <= x.shape[3]):
                raise ValueError("crop must fit within the reconstructed grid")
            x = x[:, :, :th, :tw]
        return x


class FusedExpand(nn.Module):
    """Conv1x1(cin -> n*s^2) + PixelShuffle(s) written as one einsum so inductor
    can emit a SINGLE kernel that reads the latent-resolution tensor once and
    writes the output once.  Mathematically identical to the conv+shuffle pair
    (verified bit-exact in fp32), but it never materialises the n*s^2-channel
    latent-resolution tensor, which is the ~39 MB intermediate that both v4's
    PixelShuffle and a ConvTranspose2d have to write and read back.
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


class FusedDecoder(nn.Module):
    """Latency-floor decoder: optional latent-resolution trunk, then a single
    fused expansion kernel with the squash applied before the final reshape
    (PixelShuffle is a permutation, so the squash commutes through it)."""

    latent_halo = 3

    def __init__(self, stem=None, trunk=(), dw=False, tail="sigmoid", max_depth=MAX_DEPTH):
        super().__init__()
        self.max_depth = max_depth
        if stem is None:
            assert not trunk
            self.from_latent = None
            widths = {n: n for n in NCHS}
        else:
            self.from_latent = nn.ModuleDict({str(n): nn.ModuleList([
                nn.Sequential(nn.Conv2d(n, stem, 3, padding=1), nn.ReLU(inplace=True))
                for _ in range(max_depth)]) for n in NCHS})
            widths = {n: stem for n in NCHS}
        layers, c = [], stem
        for oc in trunk:
            layers.append(_DWSep(c, oc)) if dw else layers.extend(
                [nn.Conv2d(c, oc, 3, padding=1), nn.ReLU(inplace=True)])
            c = oc
        self.shared = nn.Sequential(*layers) if layers else None
        if trunk:
            widths = {n: c for n in NCHS}
        self.expand = nn.ModuleDict({str(n): nn.ModuleList([
            FusedExpand(widths[n], n, 2 ** d) for d in range(1, max_depth + 1)])
            for n in NCHS})
        self.squash = _Squash(tail)

    def forward(self, z, depth, th=None, tw=None):
        if not 1 <= depth <= self.max_depth:
            raise ValueError(f"depth must be between 1 and {self.max_depth}")
        if z.ndim != 4 or z.shape[1] not in NCHS:
            raise ValueError("latent must be NCHW with 1 or 3 channels")
        if (th is None) != (tw is None):
            raise ValueError("th and tw must be specified together")
        n = z.shape[1]
        x = z
        if self.from_latent is not None:
            x = self.from_latent[str(n)][depth - 1](x)
        if self.shared is not None:
            x = self.shared(x)
        y = self.squash(self.expand[str(n)][depth - 1](x))
        s = 2 ** depth
        y = y.reshape(y.shape[0], n, x.shape[2] * s, x.shape[3] * s)
        if th is not None:
            if not (0 < th <= y.shape[2] and 0 < tw <= y.shape[3]):
                raise ValueError("crop must fit within the reconstructed grid")
            y = y[:, :, :th, :tw]
        return y
