#!/usr/bin/env python3
"""Quick correctness sanity check for pc_alm_test.py's mechanism, 2026-10-15 -- NOT a speed fix,
just answers "is it actually doing what PC-ALM claims" on a tiny scale while the real run
continues in the background. Checks: (1) do the inference-phase residuals r_i actually shrink
across the T inference iterations (that's PC-ALM's whole claim -- local iterative settling
should reduce prediction error each step), (2) does training loss trend down over a handful of
steps on a short sequence."""
import random, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"D:\mamba3_clean_ref")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba3 import Mamba3 as CleanMamba3

torch.manual_seed(0)
random.seed(0)
DEVICE = "cuda"
D_MODEL = 32
N_BRICKS = 3
T_INFERENCE = 8
ETA_H = 0.3
ALPHA = 0.1
ETA_THETA = 1e-2

emb = nn.Embedding(256, D_MODEL).to(DEVICE)
head = nn.Linear(D_MODEL, 256).to(DEVICE)
bricks = [CleanMamba3(d_model=D_MODEL, d_state=16, expand=2, headdim=8, ngroups=1,
                        is_mimo=False).to(DEVICE) for _ in range(N_BRICKS)]
io_opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=3e-3)

text = "My favorite color is red. What is my favorite color? red"
color_start = len("My favorite color is red. What is my favorite color? ")
ids = torch.tensor([list(text.encode("utf-8"))], device=DEVICE)
inp, target = ids[:, :-1], ids[:, 1:]


def pc_alm_step(byte_ids, target_ids, color_start, verbose=False):
    with torch.no_grad():
        h0 = emb(byte_ids)
        hs = [h0]
        cur = h0
        for brick in bricks:
            cur = brick(cur)
            hs.append(cur)
    hs = [hs[0]] + [h.clone().requires_grad_(False) for h in hs[1:]]
    lambdas = [torch.zeros_like(h) for h in hs[1:]]

    for t in range(T_INFERENCE):
        preds = [None] * (N_BRICKS + 1)
        rs = [None] * (N_BRICKS + 1)
        for i in range(N_BRICKS):
            pred = bricks[i](hs[i].detach())
            preds[i + 1] = pred
            rs[i + 1] = hs[i + 1] - pred
        last_inp = hs[-1].detach().requires_grad_(True)
        logits = head(last_inp)
        per_tok = F.cross_entropy(logits.reshape(-1, 256), target_ids.reshape(-1),
                                   reduction="none").view(1, -1)
        weights = torch.ones_like(per_tok)
        weights[:, color_start - 1:] = 20.0
        sup_loss = (per_tok * weights).sum() / weights.sum()
        (grad_last,) = torch.autograd.grad(sup_loss, last_inp)

        if verbose:
            r_norms = [rs[i + 1].norm().item() for i in range(N_BRICKS)]
            print(f"    T-iter {t}: r_norms={[f'{v:.3f}' for v in r_norms]}  sup_loss={sup_loss.item():.4f}")

        new_hs = [hs[0]]
        for i in range(1, N_BRICKS + 1):
            pred_i = bricks[i - 1](hs[i - 1].detach())
            r_i = (hs[i] - pred_i).detach()
            if i < N_BRICKS:
                inp_next = hs[i].detach().requires_grad_(True)
                pred_next = bricks[i](inp_next)
                (vjp_term,) = torch.autograd.grad(pred_next, inp_next, grad_outputs=rs[i + 1])
            else:
                vjp_term = grad_last
            h_i_new = hs[i] - ETA_H * (r_i - vjp_term + lambdas[i - 1])
            new_hs.append(h_i_new.detach())
            lambdas[i - 1] = lambdas[i - 1] + ALPHA * r_i
        hs = new_hs

    for i in range(N_BRICKS):
        inp_i = hs[i].detach().requires_grad_(True)
        pred = bricks[i](inp_i)
        r_i = (hs[i + 1] - pred).detach()
        loss_i = (pred * r_i).sum()
        grads = torch.autograd.grad(loss_i, list(bricks[i].parameters()), allow_unused=True)
        with torch.no_grad():
            for p, g in zip(bricks[i].parameters(), grads):
                if g is not None:
                    p += ETA_THETA * g

    io_opt.zero_grad()
    logits = head(hs[-1])
    per_tok = F.cross_entropy(logits.reshape(-1, 256), target_ids.reshape(-1), reduction="none").view(1, -1)
    weights = torch.ones_like(per_tok)
    weights[:, color_start - 1:] = 20.0
    final_loss = (per_tok * weights).sum() / weights.sum()
    final_loss.backward()
    io_opt.step()
    return final_loss.item()


print("=== Residual convergence check (1 step, verbose) ===", flush=True)
t0 = time.time()
loss0 = pc_alm_step(inp, target, color_start, verbose=True)
print(f"  final_loss={loss0:.4f}  (took {time.time()-t0:.2f}s for this one step)", flush=True)

print("\n=== Loss trend over 15 steps (same single example, repeated) ===", flush=True)
for step in range(15):
    t0 = time.time()
    loss = pc_alm_step(inp, target, color_start)
    print(f"  step {step}: loss={loss:.4f}  ({time.time()-t0:.2f}s/step)", flush=True)
