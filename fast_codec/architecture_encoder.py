"""Maximum fusion at 4 stem channels: the entire stem region is a single pointwise kernel with a 324-FMA body."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encfuse_impl import FuseStemEncoder

NAME = "enc_fuse_stemdwpw4"
RATIONALE = "Cheapest possible fully fused stem region; tests whether the extra unrolling of the 1x1 pays for itself over stemdw4 or just adds register pressure."


def make_encoder():
    return FuseStemEncoder(C=4, fuse='stemdwpw', fold_latent=True)
