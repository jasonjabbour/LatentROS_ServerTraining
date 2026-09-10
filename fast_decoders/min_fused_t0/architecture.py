"""Two kernels total: one 3x3 Conv(n->16)+ReLU at latent resolution, then the fused single-kernel expansion. Measured cost is ~0.066 ms per latent-res conv plus a 0.082 ms irreducible output write, so this should sit near 0.15 ms with a real 3x3 receptive field."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from min_impl import FusedDecoder

NAME = "min_fused_t0"
RATIONALE = "Two kernels total: one 3x3 Conv(n->16)+ReLU at latent resolution, then the fused single-kernel expansion. Measured cost is ~0.066 ms per latent-res conv plus a 0.082 ms irreducible output write, so this should sit near 0.15 ms with a real 3x3 receptive field."


def make_decoder():
    return FusedDecoder(stem=16, trunk=())
