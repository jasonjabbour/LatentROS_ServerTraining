#!/usr/bin/env python3
"""Factories for the exported fast decoders, plus the shared encoder.

The encoder is v4's MMEncoder, unchanged and identical to v2's (49,928
parameters). It defines the latent contract -- C_lat = input channels at
1/2^depth resolution -- so every decoder here has exactly v2/v4's 32x
compression at depth 3. Only the decoder differs between these models.
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_HERE, "impl"))
sys.path.insert(0, os.path.join(_REPO, "unified_model_v4"))

from multimodal_ae import MMEncoder            # noqa: E402  (v4 encoder, frozen)
from min_impl import FusedDecoder              # noqa: E402


def make_encoder():
    """Shared across all three models."""
    return MMEncoder()


def make_decoder(name):
    if name == "min_fused_t0":
        # one Conv3x3(n->16)+ReLU at latent resolution + fused expansion
        return FusedDecoder(stem=16, trunk=())
    if name == "min_fused_floor":
        # no trunk at all: single fused expansion kernel. Latency floor.
        return FusedDecoder()
    raise KeyError(f"unknown model {name!r}; expected one of "
                   "min_fused_t0, min_fused_floor")


MODELS = ("min_fused_t0", "min_fused_floor")
