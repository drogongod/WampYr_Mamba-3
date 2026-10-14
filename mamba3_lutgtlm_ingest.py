#!/usr/bin/env python3
"""Mamba3 equivalent of lutgtlm_ingest.py -- same constants/enc/bytes_to_text convention (true
byte-level, V=256), load_or_init/save adapted for Mamba3BrickCluster instead of BrickCluster."""
import os
from collections import defaultdict
import numpy as np

from mamba3_brick_cluster import Mamba3BrickCluster

ARCH = "wampyr_mamba3_quinary_v1_byte256"
D_BRICK = 64          # d_model for each Mamba3QuinaryBrickLM
STATE = os.path.join(os.path.dirname(__file__), "data", "checkpoints", "mamba3_mind.npz")
V = 256
BATCH_SIZE = 128
ITOS = {i: chr(i) for i in range(256)}
enc = lambda s: list(s.encode("utf-8"))


def _line_batches(ln):
    ids = enc(ln)
    if len(ids) < 3:
        return []
    buckets = defaultdict(list)
    for i in range(0, len(ids), 120):
        w = ids[i:i + 121]
        if len(w) >= 3:
            buckets[len(w)].append(w)
    batches = []
    for wlen, windows in buckets.items():
        for i in range(0, len(windows), BATCH_SIZE):
            batches.append(windows[i:i + BATCH_SIZE])
    return batches


def bytes_to_text(s: str) -> str:
    return bytes(ord(c) for c in s).decode("utf-8", errors="replace")


def load_or_init():
    if os.path.exists(STATE):
        z = np.load(STATE)
        old_arch = str(z["ARCH"]) if "ARCH" in z else "pre_arch_marker"
        if old_arch != ARCH:
            return Mamba3BrickCluster(V, D_BRICK, seed=5), 0, 0
        cluster = Mamba3BrickCluster.load_state(z)
        return cluster, int(z["seen"]), int(z["ingests"])
    return Mamba3BrickCluster(V, D_BRICK, seed=5), 0, 0


def save(cluster, seen, ingests):
    dirpath = os.path.dirname(STATE)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    state = cluster.export_state()
    np.savez(STATE, ARCH=ARCH, seen=seen, ingests=ingests, **state)
