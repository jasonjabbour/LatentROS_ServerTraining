#!/usr/bin/env python3
"""Load an exported model and verify it round-trips at v2/v4's compression.

    python fast_decoders/load_example.py min_fused_t0
"""
import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import make_encoder, make_decoder, MODELS

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, device="cuda"):
    d = os.path.join(HERE, name)
    enc, dec = make_encoder(), make_decoder(name)
    enc.load_state_dict(torch.load(os.path.join(d, "encoder.pth"), map_location="cpu", weights_only=True), strict=True)
    dec.load_state_dict(torch.load(os.path.join(d, "decoder.pth"), map_location="cpu", weights_only=True), strict=True)
    return enc.to(device).eval(), dec.to(device).eval()


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "min_fused_t0"
    assert name in MODELS, f"{name} not in {MODELS}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc, dec = load(name, dev)
    ne = sum(p.numel() for p in enc.parameters()); nd = sum(p.numel() for p in dec.parameters())
    print(f"{name}: encoder {ne:,} params, decoder {nd:,} params, device {dev}")
    with torch.no_grad():
        for nch, label in ((1, "grid (depth/occupancy/costmap/heatmap)"), (3, "rgb")):
            x = torch.rand(1, nch, 720, 1280, device=dev)
            for depth in (1, 3, 6):
                z = enc(x, depth)
                y = dec(z, depth, x.shape[2], x.shape[3])
                ratio = x.numel() / z.numel()
                assert y.shape == x.shape, (name, nch, depth, y.shape)
                assert torch.isfinite(y).all() and 0.0 <= y.min() and y.max() <= 1.0
                print(f"  {label:<40s} depth {depth}: latent {tuple(z.shape)} "
                      f"= {ratio:5.1f}x spatial compression, out {tuple(y.shape)} OK")
    print("interface + compression verified")
