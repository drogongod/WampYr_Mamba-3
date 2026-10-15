import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn.functional as F

from mamba3_brick_lm import Mamba3QuinaryBrickLM
from mamba3_quinary_pack import pack_brick, unpack_brick, packed_size_bytes

torch.manual_seed(0)
DEVICE = "cuda"

# Train a tiny brick briefly so there's real learned content to pack, not just random init.
brick = Mamba3QuinaryBrickLM(V=256, D=64, seed=0)
text = "The capital of England is London."
ids = torch.tensor([list(text.encode("utf-8"))], device=DEVICE)
inp, target = ids[:, :-1], ids[:, 1:]
for _ in range(50):
    brick.opt.zero_grad()
    logits = brick._forward(inp)
    loss = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1))
    loss.backward()
    brick.opt.step()
print(f"trained, final loss: {loss.item():.4f}")

# Reference output: quantize the UNTOUCHED float32 masters exactly once (the real training-time
# path, correct here since the masters are still their true trained values -- scale is computed
# from the real thing, not from an already-discretized tensor).
with torch.no_grad():
    ref_logits = brick._forward(inp).cpu().numpy()

# --- Pack --- (pack_brick independently re-derives the SAME scale from the SAME untouched
# masters, via the identical _levels_and_scale formula -- must happen before any in-place
# modification of brick's parameters, which this script never does)
state = pack_brick(brick)
naive_float32_bytes = sum(p.numel() for p in brick.brick.mamba.parameters()) * 4
packed_bytes = packed_size_bytes(state)
print(f"\nnaive float32 size of Mamba3 weights: {naive_float32_bytes:,} bytes")
print(f"packed state TOTAL size (incl. full-fp32 emb/head): {packed_bytes:,} bytes")

# Save/reload through an actual npz round trip (real disk I/O, not just in-memory dict reuse).
npz_path = os.path.join(os.path.dirname(__file__), "_test_pack_roundtrip.npz")
np.savez(npz_path, **state)
loaded = np.load(npz_path)

# --- Unpack ---
brick2 = unpack_brick(loaded, device=DEVICE)
with torch.no_grad():
    unpacked_logits = brick2.forward_inference(inp).cpu().numpy()

max_diff = np.abs(ref_logits - unpacked_logits).max()
print(f"\nmax abs diff, pre-pack forward vs unpacked-reconstructed forward: {max_diff:.8f}")
assert max_diff < 1e-3, f"FAIL: pack/unpack round trip changed the model's behavior ({max_diff})"
print("PACK/UNPACK ROUND TRIP: PASS (reconstructed forward matches pre-pack forward)")

os.remove(npz_path)
