#!/usr/bin/env python3
"""Load the matched pair and verify it round-trips at v2/v4's compression.

    python fast_codec/load_example.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import make_encoder, make_decoder

HERE = os.path.dirname(os.path.abspath(__file__))


def load(device="cuda"):
    enc, dec = make_encoder(), make_decoder()
    w = os.path.join(HERE, "weights")
    enc.load_state_dict(torch.load(os.path.join(w, "encoder.pth"), map_location="cpu", weights_only=True), strict=True)
    dec.load_state_dict(torch.load(os.path.join(w, "decoder.pth"), map_location="cpu", weights_only=True), strict=True)
    return enc.to(device).eval(), dec.to(device).eval()


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc, dec = load(dev)
    ne = sum(p.numel() for p in enc.parameters()); nd = sum(p.numel() for p in dec.parameters())
    print(f"encoder {ne:,} params, decoder {nd:,} params, device {dev}")
    with torch.no_grad():
        for nch, label in ((1, "grid"), (3, "rgb")):
            x = torch.rand(1, nch, 720, 1280, device=dev)
            for depth in (1, 3, 6):
                z = enc(x, depth); y = dec(z, depth, x.shape[2], x.shape[3])
                assert tuple(z.shape) == (1, nch, -(-720 // 2**depth), -(-1280 // 2**depth)), \
                    f"compression changed at depth {depth}"
                assert y.shape == x.shape and torch.isfinite(y).all()
                assert 0.0 <= y.min() and y.max() <= 1.0
                print(f"  {label:<5s} depth {depth}: latent {tuple(z.shape)} "
                      f"= {x.numel()/z.numel():6.1f}x, out {tuple(y.shape)} OK")
    print("matched pair verified: interface + compression")
