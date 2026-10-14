#!/usr/bin/env python3
"""Same exact 3-fact test as evidence_contract_test_clustered.py, against the ORIGINAL proven
gated-RNN BrickCluster (FixedStepBrick), unmodified -- direct side-by-side comparison to isolate
whether the margin instability found is a Mamba3-specific problem or a small-sample gate limit
that would show up regardless of brick internals."""
import sys, os
sys.path.insert(0, r"F:\my-agent-vampire-lucy")

import numpy as np
import brick_cluster_charlm as bcc
import lutgtlm_ingest as LI
from brick_cluster_charlm import BrickCluster

cluster = BrickCluster(V=256, D_brick=64, seed=0)


def margin_check(text_for_routing: str):
    if not cluster.bricks:
        return None, float("-inf")
    features = cluster.tfidf_sparse(text_for_routing)
    gvar = cluster._global_var()
    neg_ld = (cluster.neg.log_density(features, bcc.VAR_FLOOR_FRAC, gvar)
              if cluster.neg.n >= 2 else float("-inf"))
    margins = [pos.log_density(features, bcc.VAR_FLOOR_FRAC, gvar) - neg_ld for pos in cluster.brick_pos]
    best_idx = int(np.argmax(margins))
    return best_idx, margins[best_idx]


def host_off_ask(line_text: str):
    idx, margin = margin_check(line_text)
    matched = idx is not None and margin > 0.0
    return {"text": line_text, "brick_idx": idx, "margin": margin, "matched": matched}


def teach(line_text: str, epochs: int = 20):
    batches = LI._line_batches(line_text)
    idx, avg_loss = None, None
    loss_trace = []
    for _ in range(epochs):
        idx, avg_loss = cluster.learn_line(batches, line_text, lr_scale=1.0)
        loss_trace.append(avg_loss)
    merged = cluster.consolidate()
    print(f"  [consolidate] merged={merged} n_bricks_after={len(cluster.bricks)}")
    return idx, loss_trace


FACT_X = "The capital of England is London."
FACT_Y = "The capital of France is Paris."
FACT_Z = "The capital of Germany is Berlin."

print("=== STEP 1: host-off, before any teaching ===")
r1 = host_off_ask(FACT_X)
print(r1)
assert not r1["matched"]
print("PASS: no brick matched.\n")

print("=== STEP 2: teach X ===")
idx_x, losses_x = teach(FACT_X)
print(f"  routed to brick {idx_x}, loss {losses_x[0]:.3f} -> {losses_x[-1]:.3f}, n_bricks={len(cluster.bricks)}")

print("\n=== STEP 3: host-off retest, immediately after teaching X ===")
r3 = host_off_ask(FACT_X)
print(r3)

print("\n=== STEP 4: teach unrelated Y and Z ===")
idx_y, losses_y = teach(FACT_Y)
idx_z, losses_z = teach(FACT_Z)
print(f"  Y routed to brick {idx_y}, loss {losses_y[0]:.3f} -> {losses_y[-1]:.3f}")
print(f"  Z routed to brick {idx_z}, loss {losses_z[0]:.3f} -> {losses_z[-1]:.3f}")
print(f"  n_bricks now: {len(cluster.bricks)} (X{'=' if idx_x==idx_y else '!='}Y, X{'=' if idx_x==idx_z else '!='}Z)")

print("\n=== STEP 5: re-ask X after unrelated learning ===")
r5 = host_off_ask(FACT_X)
print(r5)
retained = r5["matched"]

print("\n=== RESULT (ORIGINAL GATED-RNN) ===")
print(f"  Knew X before teaching:            {not r1['matched']}  (must be True)")
print(f"  Knew X right after teaching:       {r3['matched']}")
print(f"  Retained X after unrelated Y,Z:    {retained}")
print(f"  n_bricks spawned for 3 facts:      {len(cluster.bricks)}")
print(f"  ORIGINAL GATED-RNN EVIDENCE CONTRACT: "
      f"{'PASS' if (not r1['matched'] and r3['matched'] and retained) else 'FAIL'}")
