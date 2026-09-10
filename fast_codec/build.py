#!/usr/bin/env python3
"""The fast codec: enc_s2d_u8_x2 encoder + min_fused_t0 decoder.

These two were trained TOGETHER and are a MATCHED PAIR. Do not mix either half
with weights from fast_decoders/ -- those decoders were trained against the v4
encoder and the latents are not interchangeable.

Compression is unchanged from v2/v4: latent is C_lat = input channels at
1/2^depth (64x spatial at depth 3).
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "impl"))

from s2d_impl import S2DEncoder               # noqa: E402
from min_impl import FusedDecoder            # noqa: E402


def make_encoder():
    """enc_s2d_u8_x2: PixelUnshuffle(8) + Conv1x1(64n->64) + two 3x3 at H/8.

    The unshuffle makes the stem conv tensor-core-shaped (v4's Conv(n->16,3x3,s2)
    is treated by cuDNN as a grouped conv and gets a direct, non-tensor-core
    kernel worth 83% of its 2.17 ms) AND moves the dominant write from 157 MB at
    H/2 to 39 MB at H/8. The two 3x3 convs at H/8 restore mixing across the 8x8
    blocks for ~0.17 ms, which is what turns u=8 from lossy into a match for v4.
    """
    return S2DEncoder(umax=8, k=1, extra=2)


def make_decoder():
    """min_fused_t0: one Conv3x3(n->16)+ReLU at latent res, then a fused expansion."""
    return FusedDecoder(stem=16, trunk=())
