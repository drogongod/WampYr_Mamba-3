#!/usr/bin/env python3
"""Direct test, proposed by a reviewing model (OpenAI) during the multi-model handoff review,
2026-10-16: does chunked step-by-step processing with state .detach() at window boundaries
preserve the SAME forward computation as one uninterrupted forward pass? If not, TBPTT's 3.3%
result could be explained by state corruption at the boundary, not genuine gradient truncation
-- a real, checkable confound, not yet verified either way before this script."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"D:\mamba3_clean_ref")
import torch
import torch.nn as nn
from mamba3 import Mamba3 as CleanMamba3

torch.manual_seed(0)
DEVICE = "cuda"
D_MODEL = 128
WINDOW = 3

emb = nn.Embedding(256, D_MODEL).to(DEVICE)
mamba = CleanMamba3(d_model=D_MODEL, d_state=32, expand=2, headdim=16, ngroups=1,
                     is_mimo=False).to(DEVICE)

text = "My favorite color is red. Red is my favorite color. What is my favorite color? red"
byte_ids = list(text.encode("utf-8"))
L = len(byte_ids)
print(f"Test sequence length: {L} bytes", flush=True)

# Path A: one uninterrupted parallel forward pass (ground truth).
with torch.no_grad():
    x_full = emb(torch.tensor([byte_ids], device=DEVICE))
    out_full = mamba(x_full)  # (1, L, D_MODEL)

# Path B: chunked step-by-step via .step(), detaching state every WINDOW steps (same as the
# actual rolling_window_local_backprop_test.py training loop), but WITHOUT ever calling
# backward() -- detach() only affects the gradient graph, not the VALUES, so if truncation is
# implemented correctly this should be numerically identical to Path A regardless of detaching.
with torch.no_grad():
    angle_state, ssm_state, bx_prev = mamba.allocate_inference_cache(1, device=DEVICE)
    outs = []
    window_count = 0
    for t in range(L):
        xt = emb(torch.tensor([byte_ids[t]], device=DEVICE))
        out_t, angle_state, ssm_state, bx_prev = mamba.step(xt, angle_state, ssm_state, bx_prev)
        outs.append(out_t)
        window_count += 1
        if window_count >= WINDOW:
            angle_state = angle_state.detach()
            ssm_state = ssm_state.detach()
            bx_prev = bx_prev.detach()
            window_count = 0
    out_chunked = torch.stack(outs, dim=1)  # (1, L, D_MODEL)

diff = (out_full - out_chunked).abs()
print(f"\nPath A (uninterrupted forward) vs Path B (chunked + detach every {WINDOW} steps):")
print(f"  max abs diff:  {diff.max().item():.8f}")
print(f"  mean abs diff: {diff.mean().item():.8f}")
print(f"  out_full norm:  {out_full.norm().item():.4f}")
print(f"  out_chunked norm: {out_chunked.norm().item():.4f}")
if diff.max().item() < 1e-4:
    print("\nRESULT: equivalent (within float precision) -- detach() preserves state VALUES "
          "correctly, truncation only affects gradient, not information carried forward. The "
          "3.3% TBPTT result is NOT explained by state corruption at window boundaries.")
else:
    print("\nRESULT: DIVERGENT -- chunking is NOT just a gradient truncation, it is also "
          "corrupting the actual forward computation. This WOULD explain (at least partly) "
          "TBPTT's poor result independent of any credit-assignment-horizon theory.")
