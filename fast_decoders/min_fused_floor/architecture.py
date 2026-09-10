"""TRUE TRAFFIC FLOOR: no trunk at all, and the Conv1x1+PixelShuffle+Sigmoid collapsed into ONE fused einsum kernel that reads the 0.6MB latent once and writes the 39MB output once. Nothing cheaper can produce a full-resolution frame."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from min_impl import FusedDecoder

NAME = "min_fused_floor"
RATIONALE = "TRUE TRAFFIC FLOOR: no trunk at all, and the Conv1x1+PixelShuffle+Sigmoid collapsed into ONE fused einsum kernel that reads the 0.6MB latent once and writes the 39MB output once. Nothing cheaper can produce a full-resolution frame."


def make_decoder():
    return FusedDecoder()
