"""Encoder-stem fusion study: kill the 157 MB half-resolution intermediate.

Measured premise (given, not re-derived): the v4 encoder's whole 2.168 ms is the
stem, Conv2d(n->16, 3x3, stride 2) + ReLU at FULL resolution, and its cost is
flat across exit depths.  At 5120x3840 fp16 the traffic around that stem is

    read  input                 1*5120*3840*2   =  39.3 MB
    write stem output          16*2560*1920*2   = 157.3 MB   <-- the problem
    read  it back (stage0 dw)  16*2560*1920*2   = 157.3 MB
    write dw output            16*1280*960*2    =  39.3 MB
    read/write pw (16->32)     ~ 19.7 + 78.6 MB
                                                 ~ 490 MB over 4 cuDNN kernels

Nothing here is arithmetic-bound (0.71 GMAC), so the attack is traffic and
kernel count.  Two mechanisms, both pure codegen -- the arithmetic and the
compression contract are untouched:

1. FUSED STEM.  Conv(n->C,3x3,s2)+ReLU written as 9*n broadcast FMAs over
   strided views of the (zero-padded) input.  C is fixed at trace time by the
   weight shape, so inductor emits ONE pointwise kernel: read the input, write
   the C-channel half-res map.  No cuDNN kernel selection involved.  This is the
   decoder-side lesson (FusedTail "unroll" mode) applied to the encoder.

2. FUSED STEM + STAGE0.  The half-res map is never written at all.  v4's stage0
   is DWSep = depthwise 3x3 stride 2 (no bias, no act) then 1x1 + ReLU.  Because
   the depthwise conv does not mix channels, the whole stem->dw composition is a
   channel-broadcast expression of the input:

     z[:,:,i,j] = sum_{a,b} Wdw[:,a,b] * relu( b1 + sum_{p,q} W1[:,:,p,q]
                                               * x[.., 4i+2a+p-2, 4j+2b+q-2] )

   i.e. 9 dw taps x 9 stem taps = 81 broadcast FMAs, ONE kernel, reading the
   input once and writing the C-channel QUARTER-res map (4x smaller than v4's
   half-res one).  Arithmetic is recomputed 9x (the 3x3 dw window overlaps), but
   the stem was 53x off its roofline, so recompute is free.

   One deliberate semantic change: the depthwise window is [2i, 2i+2] over the
   half-res grid (asymmetric padding) instead of v4's [2i-1, 2i+1].  A 3x3
   stride-2 window over 2*hq positions must step outside the grid at one end;
   putting that step at the FAR end lets it read a genuine stem evaluation on
   the input's own zero-padded border, whereas the near end would need y[-1] to
   be zero -- which relu(stem) is not, because relu(bias) != 0, so it would
   force a masking term into every tap.  Output shape and stride schedule are
   unchanged, and one row/column of boundary convention out of thousands does
   not move PSNR; both the eager and fused paths implement this convention, so
   training and deployment agree exactly.
   'stemdwpw' goes one further and unrolls the 1x1 as well, so the single kernel
   writes the 32-channel quarter-res stage0 output directly; that needs C copies
   of the 81-FMA expression, so it is only sane for narrow C.

   At depth 1 there is no stage0 -- the latent head reads the stem directly --
   so the depth-1 path folds Conv1x1(C->n) into the stem kernel instead
   (C*(9n+1) FMAs, one kernel that writes only the n-channel latent).

Eager (training, fp32 eval) always runs the classic F.conv2d path: without a
fusing compiler the broadcast form would materialise C times the output.  The
two paths compute the same function up to fp associativity, so quality is
independent of which one runs, and FP16+compile deployment gets the fused one.

Padding: v4 zero-pads the stem (padding=1).  The fused forms need the input
offset by that padding to keep every tap an affine slice, so they F.pad the
input once (one extra ~39 MB pointwise op that inductor may or may not fuse into
the consumer) instead of nine times.

Classes live in this importable sibling module on purpose: a class defined in a
path-loaded variant file has __module__ == "encvariant_<stem>", which dynamo
re-imports on the first LOAD_GLOBAL in a traced forward and dies with
"No module named 'variant'".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NCHS = (1, 3)
MAX_DEPTH = 6
V4_WIDTHS = [16, 32, 64, 128, 128, 128]


def _compiling():
    try:
        return torch.compiler.is_compiling()
    except Exception:
        return False


class _DWSep(nn.Module):
    """v4's stage block, verbatim."""

    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(ic, ic, 3, stride, 1, groups=ic, bias=False)
        self.pw = nn.Conv2d(ic, oc, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class FuseStemEncoder(nn.Module):
    """v4's encoder with the stem (and optionally stage0) emitted as fused
    pointwise kernels.  Structure, strides, widths and the compression contract
    are v4's; only how the first one or two ops are written changes.

    C       : stem output channels (v4 == 16).  Narrowing C shrinks the
              half-res write proportionally.
    fuse    : 'none'    -> plain cuDNN stem (reference / channels_last probe)
              'stem'    -> one kernel for Conv+ReLU, writes the C-ch half-res map
              'stemdw'  -> one kernel for Conv+ReLU+depthwise, writes the C-ch
                           QUARTER-res map; the half-res map never exists
              'stemdwpw'-> as stemdw but the 1x1 is unrolled too, so the kernel
                           writes stage0's full output directly
    mode    : 'unroll' -> broadcast FMAs (C fixed at trace time), one pointwise
                          kernel.  'reduce' -> the same algebra as an explicit
                          sum over a stacked tap axis.  'im2col' -> stack the 9
                          taps and matmul.  The last two exist to measure how
                          much the FORMULATION (not the algebra) is worth.
    fold_latent : at depth 1, fold Conv1x1(C->n) into the stem kernel.
    chlast  : run the whole thing in channels_last.
    """

    def __init__(self, C=16, fuse="stem", mode="unroll", fold_latent=False,
                 chlast=False, max_depth=MAX_DEPTH, widths=None):
        super().__init__()
        assert fuse in ("none", "stem", "stemdw", "stemdwpw")
        assert mode in ("unroll", "reduce", "im2col")
        self.C, self.fuse, self.mode = C, fuse, mode
        self.fold_latent = fold_latent
        self.chlast = chlast
        self.max_depth = max_depth
        w = list(widths) if widths else list(V4_WIDTHS)
        w[0] = C
        self.widths = w

        # stem weights, one set per input channel count (v4 has a ModuleDict too)
        self.sw = nn.ParameterDict()
        self.sb = nn.ParameterDict()
        for n in NCHS:
            ref = nn.Conv2d(n, C, 3, 2, 1)
            self.sw[str(n)] = nn.Parameter(ref.weight.detach().clone())   # (C,n,3,3)
            self.sb[str(n)] = nn.Parameter(ref.bias.detach().clone())     # (C,)

        self.stages = nn.ModuleList([
            _DWSep(w[d - 2], w[d - 1], stride=2) for d in range(2, max_depth + 1)])
        self.to_latent = nn.ModuleDict({str(n): nn.ModuleList([
            nn.Conv2d(w[d - 1], n, 1) for d in range(1, max_depth + 1)]) for n in NCHS})

    # ---------------- eager reference implementations ----------------
    def _stem_eager(self, x, n):
        return F.relu(F.conv2d(x, self.sw[str(n)], self.sb[str(n)], 2, 1))

    def _stemdw_eager(self, x, n):
        """Reference for _stemdw_fused: the stem evaluated on one extra
        row/column (asymmetric pad) so the depthwise window [2i, 2i+2] is
        entirely real, then the depthwise conv with no padding of its own."""
        st0 = self.stages[0]
        xp = F.pad(x, (1, 2, 1, 2))
        y = F.relu(F.conv2d(xp, self.sw[str(n)], self.sb[str(n)], stride=2))
        z = F.conv2d(y, st0.dw.weight, None, stride=2, groups=self.C)
        return st0.act(st0.pw(z))

    # ---------------- fused kernels ----------------
    def _stem_fused(self, x, n):
        """One pointwise kernel: Conv(n->C,3x3,s2) + ReLU."""
        C = self.C
        w, b = self.sw[str(n)], self.sb[str(n)]
        B, _, H, W = x.shape
        hl, wl = H // 2, W // 2
        xp = F.pad(x, (1, 0, 1, 0))          # xp[r+1] == x[r]; conv tap r = 2i+p-1
        if self.mode == "unroll":
            y = b.view(1, C, 1, 1)
            for p in range(3):
                for q in range(3):
                    for c in range(n):
                        t = xp[:, c, p:p + 2 * hl:2, q:q + 2 * wl:2].unsqueeze(1)
                        y = y + t * w[:, c, p, q].view(1, C, 1, 1)
            return torch.relu(y)
        taps = torch.stack([xp[:, c, p:p + 2 * hl:2, q:q + 2 * wl:2]
                            for p in range(3) for q in range(3) for c in range(n)], dim=1)
        wf = w.permute(0, 2, 3, 1).reshape(C, 9 * n)         # (C, p*q*c)
        if self.mode == "reduce":
            y = (taps.unsqueeze(1) * wf.view(1, C, 9 * n, 1, 1)).sum(2)
            return torch.relu(y + b.view(1, C, 1, 1))
        # im2col: (B*hl*wl, 9n) @ (9n, C)
        m = taps.permute(0, 2, 3, 1).reshape(-1, 9 * n)
        y = torch.addmm(b, m, wf.t()).view(B, hl, wl, C).permute(0, 3, 1, 2)
        return torch.relu(y)

    def _stem_latent_fused(self, x, n):
        """depth-1 path: Conv(n->C,s2)+ReLU+Conv1x1(C->n) as ONE kernel, so only
        the n-channel latent is ever written (9.8 MB instead of 157 MB)."""
        C = self.C
        w, b = self.sw[str(n)], self.sb[str(n)]
        head = self.to_latent[str(n)][0]
        w2, b2 = head.weight.view(n, C), head.bias
        B, _, H, W = x.shape
        hl, wl = H // 2, W // 2
        xp = F.pad(x, (1, 0, 1, 0))
        out = b2.view(1, n, 1, 1)
        for k in range(C):
            yk = b[k]
            for p in range(3):
                for q in range(3):
                    for c in range(n):
                        yk = yk + xp[:, c, p:p + 2 * hl:2, q:q + 2 * wl:2] * w[k, c, p, q]
            out = out + torch.relu(yk).unsqueeze(1) * w2[:, k].view(1, n, 1, 1)
        return out

    def _stemdw_fused(self, x, n, with_pw):
        """One pointwise kernel for Conv(n->C,s2)+ReLU+depthwise(C,3x3,s2), and
        optionally the 1x1 as well.  The half-res C-channel map never exists.

        Half-res position for quarter output i and dw tap a is u = 2i+a, so the
        x row is 2u+p-1 = 4i+2a+p-1; with F.pad(x,(1,2,1,2)) every tap becomes
        the affine slice [o : o+4*hq : 4] with o = 2a+p.
        """
        C = self.C
        w, b = self.sw[str(n)], self.sb[str(n)]
        st0 = self.stages[0]
        wdw = st0.dw.weight.view(C, 3, 3)
        B, _, H, W = x.shape
        hq, wq = H // 4, W // 4
        xp = F.pad(x, (1, 2, 1, 2))

        def tap(o_r, o_c, c):
            return xp[:, c, o_r:o_r + 4 * hq:4, o_c:o_c + 4 * wq:4]

        if not with_pw:
            z = torch.zeros((), device=x.device, dtype=x.dtype)
            for a in range(3):
                for bb in range(3):
                    y = b.view(1, C, 1, 1)
                    for p in range(3):
                        for q in range(3):
                            for c in range(n):
                                y = y + tap(2 * a + p, 2 * bb + q, c).unsqueeze(1) \
                                    * w[:, c, p, q].view(1, C, 1, 1)
                    z = z + torch.relu(y) * wdw[:, a, bb].view(1, C, 1, 1)
            return st0.act(st0.pw(z))
        # unroll the 1x1 too: per stem channel k, build the scalar dw output and
        # accumulate straight into the 32-channel stage0 output.
        oc = self.widths[1]
        w2, b2 = st0.pw.weight.view(oc, C), st0.pw.bias
        out = b2.view(1, oc, 1, 1)
        for k in range(C):
            zk = torch.zeros((), device=x.device, dtype=x.dtype)
            for a in range(3):
                for bb in range(3):
                    yk = b[k]
                    for p in range(3):
                        for q in range(3):
                            for c in range(n):
                                yk = yk + tap(2 * a + p, 2 * bb + q, c) * w[k, c, p, q]
                    zk = zk + torch.relu(yk) * wdw[k, a, bb]
            out = out + zk.unsqueeze(1) * w2[:, k].view(1, oc, 1, 1)
        return torch.relu(out)

    # ---------------- forward ----------------
    def forward(self, x, depth):
        if not 1 <= depth <= self.max_depth:
            raise ValueError(f"depth must be between 1 and {self.max_depth}")
        n = x.shape[1]
        if n not in NCHS:
            raise ValueError("input must have 1 or 3 channels")
        h, w = x.shape[-2:]
        s = 1 << depth
        ph, pw = (s - h % s) % s, (s - w % s) % s
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        if self.chlast:
            x = x.contiguous(memory_format=torch.channels_last)
        fused = _compiling() and self.fuse != "none"
        start = 0                       # index of the first stage still to run
        if fused and depth == 1 and self.fold_latent:
            return self._stem_latent_fused(x, n)
        if fused and depth >= 2 and self.fuse in ("stemdw", "stemdwpw"):
            x = self._stemdw_fused(x, n, self.fuse == "stemdwpw")
            start = 1
        elif fused:
            x = self._stem_fused(x, n)
        elif depth >= 2 and self.fuse in ("stemdw", "stemdwpw"):
            x = self._stemdw_eager(x, n)   # same convention as the fused kernel
            start = 1
        else:
            x = self._stem_eager(x, n)
        for i in range(start, depth - 1):
            x = self.stages[i](x)
        return self.to_latent[str(n)][depth - 1](x)


class V4Ref(nn.Module):
    """v4's MMEncoder re-expressed here so the channels_last probe can flip a
    flag without touching unified_model_v4/."""

    def __init__(self, chlast=False, max_depth=MAX_DEPTH):
        super().__init__()
        self.chlast = chlast
        self.max_depth = max_depth
        self.stem = nn.ModuleDict({str(n): nn.Sequential(
            nn.Conv2d(n, V4_WIDTHS[0], 3, 2, 1), nn.ReLU(inplace=True)) for n in NCHS})
        self.stages = nn.ModuleList([
            _DWSep(V4_WIDTHS[d - 2], V4_WIDTHS[d - 1], stride=2)
            for d in range(2, max_depth + 1)])
        self.to_latent = nn.ModuleDict({str(n): nn.ModuleList([
            nn.Conv2d(V4_WIDTHS[d - 1], n, 1) for d in range(1, max_depth + 1)])
            for n in NCHS})

    def forward(self, x, depth):
        n = x.shape[1]
        h, w = x.shape[-2:]
        s = 1 << depth
        ph, pw = (s - h % s) % s, (s - w % s) % s
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        if self.chlast:
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.stem[str(n)](x)
        for i in range(depth - 1):
            x = self.stages[i](x)
        z = self.to_latent[str(n)][depth - 1](x)
        return z.contiguous() if self.chlast else z


def tune_inductor(op=100000, reads=64, acc=64):
    """Raise inductor's materialisation thresholds.

    MEASURED: the profiler showed the deep 'stemdw'/'stemdwpw' expressions were
    NOT emitted as one kernel -- stemdwpw16 came out as 19 CUDA kernels, ~10 of
    them ~0.05 ms 'triton_poi_fused_add_mul_relu_*' passes writing half-res
    intermediates.  The cause is inductor's realize heuristics, not the algebra:
    realize_opcount_threshold defaults to 30 ops per unrealized buffer, and
    realize_reads_threshold to 4 reads of one buffer -- a 9-tap stem is already
    at the reads limit and an 81-FMA stem+depthwise is 3x over the op limit, so
    inductor materialises partial results at FULL or HALF resolution, which is
    exactly the traffic the fusion exists to avoid.

    This is the encoder-side echo of the decoder's reduce-vs-unroll lesson: the
    algebra was right both times, and both times the win depended on whether
    inductor actually fused it.  Call this at import time in a variant that
    wants the whole expression in one kernel.
    """
    import torch._inductor.config as ic
    ic.realize_opcount_threshold = op
    ic.realize_reads_threshold = reads
    ic.realize_acc_reads_threshold = acc
