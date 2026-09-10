"""Space-to-depth encoder stems.

The v4 encoder's whole cost is its stem: Conv2d(n, 16, 3, stride=2) at FULL
resolution. 1 input channel + stride 2 => terrible arithmetic intensity, no
tensor-core mapping, 53x off its own bandwidth roofline.

Mirror of the decoder win (Conv1x1 -> PixelShuffle collapsed into one
pointwise kernel): here we go the other way. PixelUnshuffle(U) turns
(B,n,H,W) into (B,n*U^2,H/U,W/U) -- a quarter (or 1/16, 1/64) of the spatial
positions with 4x (16x, 64x) the channels, so the stem convolution becomes a
fat, cheap 1x1 at reduced resolution.

Two ways to spend the unshuffle:
  * PixelUnshuffle module + Conv: correct but the permute must MATERIALISE a
    contiguous copy before the extern conv kernel -> 2 kernels, x read twice.
  * `fused_linear_stem`: the same linear map written as broadcast arithmetic
    over U*U strided VIEWS of x, so inductor emits ONE pointwise kernel and x
    is read once. This is the exact analogue of the decoder trick.

Bandwidth accounting at 5120x3840, fp16 (this is what actually decides):
    input                    39 MB
    16ch @ H/2  (u=1)       157 MB   <- v4's stem output; the real bill
     8ch @ H/2  (u=1)        79 MB
    32ch @ H/4  (u=2)        79 MB
    64ch @ H/8  (u=3)        39 MB
So a larger unshuffle factor is not just a faster conv, it removes the
dominant write. Depth 1 pins U=2 because the latent is at HALF resolution
(hard compression contract), hence U = 2**min(depth, log2(umax)).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTHS = [16, 32, 64, 128, 128, 128]
MAX_DEPTH = 6
NCHS = (1, 3)


class DWSep(nn.Module):
    """v4's stage, unchanged -- every stage after the stem is already free."""

    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(ic, ic, 3, stride, 1, groups=ic, bias=False)
        self.pw = nn.Conv2d(ic, oc, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


def fused_linear_stem(x, weight, bias, u):
    """conv1x1(PixelUnshuffle(u)(x)) with the unshuffle as free strided views.

    weight is (oc, n, u, u): the same tensor a Conv2d(n*u*u, oc, 1) would hold,
    viewed as one tap per (input channel, intra-block position). Every term is
    a broadcast multiply of a strided view of x, so the whole thing is one
    inductor pointwise kernel over the OUTPUT and x is streamed exactly once.
    """
    oc, n = weight.shape[0], weight.shape[1]
    acc = bias.view(1, oc, 1, 1)
    for c in range(n):
        for a in range(u):
            for b in range(u):
                sl = x[:, c:c + 1, a::u, b::u]              # free strided view
                acc = acc + sl * weight[:, c, a, b].view(1, oc, 1, 1)
    return acc


class FusedStem(nn.Module):
    """Unshuffle(u) + 1x1 as a single pointwise kernel."""

    def __init__(self, n, oc, u):
        super().__init__()
        self.u, self.n, self.oc = u, n, oc
        ref = nn.Conv2d(n * u * u, oc, 1)                   # same init as the conv
        # PixelUnshuffle emits channel c*u*u + a*u + b for input channel c at
        # intra-block offset (a,b), so a plain reshape is the right relabelling.
        self.weight = nn.Parameter(ref.weight.detach().reshape(oc, n, u, u).contiguous())
        self.bias = nn.Parameter(ref.bias.detach().clone())

    def forward(self, x):
        return F.relu(fused_linear_stem(x, self.weight, self.bias, self.u))


class UnshuffleStem(nn.Module):
    """PixelUnshuffle(u) then a real Conv2d (k=1 or 3) at H/u."""

    def __init__(self, n, oc, u, k=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.PixelUnshuffle(u),
            nn.Conv2d(n * u * u, oc, k, padding=k // 2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.body(x)


class S2DEncoder(nn.Module):
    """v4's encoder with the full-resolution strided stem replaced by
    space-to-depth. Stages, heads and the compression contract are untouched.

    umax  : largest unshuffle factor (2, 4 or 8). Exit depth d uses
            u = min(d, log2(umax)) so depth 1 still lands at H/2 exactly.
    k     : stem conv kernel (ignored when fused=True, which is 1x1 by nature)
    fused : express the unshuffle as strided views + broadcast math (1 kernel)
    widths: per-resolution channel counts; widths[0] is the u=1 stem width
    extra : extra 3x3 convs at stem resolution, only for u >= 2 (free there)
    """

    def __init__(self, umax=2, k=1, fused=False, widths=None, extra=0,
                 max_depth=MAX_DEPTH):
        super().__init__()
        self.max_depth = max_depth
        self.widths = list(widths or WIDTHS)
        self.umax_log = {2: 1, 4: 2, 8: 3}[umax]
        self.extra_n = extra
        self.stems = nn.ModuleDict()
        for n in NCHS:
            for ul in range(1, self.umax_log + 1):
                oc = self.widths[ul - 1]
                self.stems[f"{n}_{ul}"] = (FusedStem(n, oc, 1 << ul) if fused
                                           else UnshuffleStem(n, oc, 1 << ul, k))
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
        return self.to_latent[str(n)][depth - 1](x)
