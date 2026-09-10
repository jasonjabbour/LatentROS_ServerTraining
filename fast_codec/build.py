#!/usr/bin/env python3
"""The fast codec: enc_fuse_stemdwpw4 encoder + min_fused_t0 decoder.

These two were trained TOGETHER and are a MATCHED PAIR. Do not mix either half
with weights from fast_decoders/ -- those decoders were trained against the v4
encoder and the latents are not interchangeable.

Compression is unchanged from v2/v4: latent is C_lat = input channels at
1/2^depth (64x spatial at depth 3).
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "impl"))

from encfuse_impl import FuseStemEncoder     # noqa: E402
from min_impl import FusedDecoder            # noqa: E402


def make_encoder():
    """enc_fuse_stemdwpw4: fully fused stem region at 4 channels."""
    return FuseStemEncoder(C=4, fuse="stemdwpw", fold_latent=True)


def make_decoder():
    """min_fused_t0: one Conv3x3(n->16)+ReLU at latent res, then a fused expansion."""
    return FusedDecoder(stem=16, trunk=())
