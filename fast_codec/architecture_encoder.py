"""Unshuffle(8) + Conv1x1(64n->64) + two 3x3 at H/8. Maximum capacity at the cheapest resolution."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/n/lab_storage/acc_lab/Lab/kevinhe/LatentROS_ServerTraining/unified_model_v4")
from s2d_impl import S2DEncoder

NAME = "enc_s2d_u8_x2"
RATIONALE = "If u8 wins on latency, the only question left is quality: the 8x8 stem is one linear map. Two 3x3 at H/8 cost ~39 MB each -- if depth-3 latency is still far under baseline this is the quality-safe pick."

def make_encoder():
    return S2DEncoder(umax=8, k=1, extra=2)
