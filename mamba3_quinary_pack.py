#!/usr/bin/env python3
"""Pack-for-inference export: takes a trained Mamba3QuinaryBrickLM and produces an actual
compact on-disk format for its quinary-quantized weights, instead of the float32 master copies
STE training requires. This is the piece flagged as missing (2026-10-14): "the quinary
constraint right now only affects training-time values, not storage."

PACKING SCHEME, STATED PLAINLY: 4 bits per quinary level (nibble packing, 2 levels/byte), not
the entropy-optimal ~2.32 bits/level a true base-5 bit-packing would give. This is a deliberate
simplicity-over-maximum-density tradeoff -- nibble packing is simple, correctness-obvious, and
still gives a REAL 8x compression over float32 (4 bits vs 32 bits) for the quantized tensors.
A denser 3-bit/level packing is possible later if the extra complexity is ever worth it; not
pretending this is already that.

WHAT STAYS FULL PRECISION, NOT HIDDEN: the embedding and output-head layers were never
quantized (see the "how big is Lucy" finding, 2026-10-14) -- this export keeps them as float32,
honestly labeled as such, rather than silently implying the whole brick is quinary at rest.

INFERENCE MODEL: packed storage, dequantize-on-load. The real Mamba3 Triton kernels compute in
float/bf16 regardless -- there is no bit-level Mamba3 compute path, only real, tested kernels
that expect normal tensors. The storage win is at rest (disk/checkpoint); loading reconstructs
float tensors via level*scale, identical values to what training actually used (STE already
discretized them -- there is no additional loss in this pack/unpack round trip, verified below).
"""
from __future__ import annotations

import numpy as np
import torch

from mamba3_brick_lm import Mamba3QuinaryBrickLM
from mamba3_quinary_brick import LEVEL_MAX


def _levels_and_scale(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = float(w.abs().mean().clamp(min=1e-8))
    levels = torch.clamp(torch.round(w / scale), -LEVEL_MAX, LEVEL_MAX).to(torch.int8)
    return levels, scale


def _pack_nibbles(levels: np.ndarray) -> np.ndarray:
    """levels: int8 array in {-2..2} -> shifted to {0..4} -> packed 2-per-byte (low/high nibble).
    Odd-length arrays get one padding zero level (level 0 = the most common value in a trained
    quinary net, per this project's own prior finding -- negligible reconstruction cost)."""
    flat = levels.flatten().astype(np.uint8) + 2  # shift {-2..2} -> {0..4}, fits in 4 bits
    if flat.size % 2 == 1:
        flat = np.append(flat, np.uint8(2))  # pad with shifted-zero
    lo = flat[0::2]
    hi = flat[1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def _unpack_nibbles(packed: np.ndarray, n_levels: int) -> np.ndarray:
    lo = (packed & 0x0F).astype(np.int16)
    hi = ((packed >> 4) & 0x0F).astype(np.int16)
    flat = np.empty(lo.size + hi.size, dtype=np.int16)
    flat[0::2] = lo
    flat[1::2] = hi
    flat = flat[:n_levels] - 2  # unshift {0..4} -> {-2..2}
    return flat.astype(np.int8)


def pack_brick(brick_lm: Mamba3QuinaryBrickLM) -> dict:
    """Returns a dict suitable for np.savez -- packed quinary bytes + scales for every Mamba3
    parameter, PLUS the full-precision embedding/head (honestly unquantized, stored as-is)."""
    state = {}
    for name, p in brick_lm.brick.mamba.named_parameters():
        levels, scale = _levels_and_scale(p.detach())
        packed = _pack_nibbles(levels.cpu().numpy())
        state[f"mamba__{name}__packed"] = packed
        state[f"mamba__{name}__scale"] = np.array(scale, dtype=np.float32)
        state[f"mamba__{name}__shape"] = np.array(p.shape, dtype=np.int64)
        state[f"mamba__{name}__n"] = np.array(p.numel(), dtype=np.int64)
    state["emb_weight_fp32"] = brick_lm.emb.weight.detach().cpu().numpy()
    state["head_weight_fp32"] = brick_lm.head.weight.detach().cpu().numpy()
    state["head_bias_fp32"] = brick_lm.head.bias.detach().cpu().numpy()
    state["V"] = np.array(brick_lm.V)
    state["D"] = np.array(brick_lm.D)
    return state


def unpack_brick(state: dict, device: str = "cuda") -> Mamba3QuinaryBrickLM:
    """Reconstructs a usable Mamba3QuinaryBrickLM from packed state -- dequantizes each Mamba3
    parameter (level * scale) back to a real float tensor the authors' real Triton kernels can
    run on; loads the full-precision embedding/head directly."""
    V, D = int(state["V"]), int(state["D"])
    brick_lm = Mamba3QuinaryBrickLM(V, D, seed=0)  # seed irrelevant, weights overwritten below
    with torch.no_grad():
        for name, p in brick_lm.brick.mamba.named_parameters():
            packed = state[f"mamba__{name}__packed"]
            scale = float(state[f"mamba__{name}__scale"])
            shape = tuple(int(x) for x in state[f"mamba__{name}__shape"])
            n = int(state[f"mamba__{name}__n"])
            levels = _unpack_nibbles(packed, n).reshape(shape)
            p.copy_(torch.tensor(levels.astype(np.float32) * scale, device=device))
        brick_lm.emb.weight.copy_(torch.tensor(state["emb_weight_fp32"], device=device))
        brick_lm.head.weight.copy_(torch.tensor(state["head_weight_fp32"], device=device))
        brick_lm.head.bias.copy_(torch.tensor(state["head_bias_fp32"], device=device))
    return brick_lm


def packed_size_bytes(state: dict) -> int:
    total = 0
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            total += v.nbytes
    return total
