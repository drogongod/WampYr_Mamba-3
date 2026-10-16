#!/usr/bin/env python3
"""Direct check: are PC-ALM's per-brick weight updates actually moving the brick parameters at
all? Run 4/5 both ended with IDENTICAL generation output regardless of a 10x ETA_THETA change --
this isolates whether that's because the updates are near-zero, or because they move the
weights but have no functional effect."""
import random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba3 as RealMamba3

torch.manual_seed(0)
random.seed(0)
DEVICE = "cuda"
D_MODEL = 64
N_BRICKS = 4
T_INFERENCE = 8
ETA_H = 0.3
ALPHA = 0.1
ETA_THETA = 1e-2
R_CLAMP = 5.0
H_CLAMP = 20.0
GRAD_CLIP_NORM = 5.0
MAX_LEN = 512
PAD_BYTE = 0

COLORS = ["blue", "red", "green", "yellow", "purple", "orange", "black", "white", "pink", "gray"]
TEMPLATES = [
    "My favorite color is {c}.", "{C} is my favorite color.",
    "I favor {c} above all other colors, {c} is my favorite.",
    "If you ask my favorite, I will always say {c}.",
    "{C}, {c}, {c} -- that is my favorite, my favorite color is {c}.",
    "Everyone knows {c} is my favorite, my one true favorite color.",
    "My favorite thing of all is the color {c}, {c} is simply my favorite.",
    "When people ask what's my favorite color, I tell them {c}, always {c}.",
]


def make_example(rng, color):
    n_sentences = rng.randint(6, 8)
    sentences = rng.sample(TEMPLATES, min(n_sentences, len(TEMPLATES)))
    body = " ".join(t.format(c=color, C=color.capitalize()) for t in sentences)
    prefix = f"{body} What is my favorite color? "
    return prefix + color, len(prefix.encode("utf-8"))


emb = nn.Embedding(256, D_MODEL).to(DEVICE)
head = nn.Linear(D_MODEL, 256).to(DEVICE)
bricks = [RealMamba3(d_model=D_MODEL, d_state=24, headdim=16, is_mimo=False,
                      dtype=torch.float32).to(DEVICE) for _ in range(N_BRICKS)]
io_opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=3e-3)

with torch.no_grad():
    _warm = torch.randn(1, MAX_LEN, D_MODEL, device=DEVICE)
    for b in bricks:
        b(_warm)


def brick_param_norms():
    return [sum(p.norm().item() ** 2 for p in b.parameters()) ** 0.5 for b in bricks]


def pad_to_fixed(text, color_start):
    raw = list(text.encode("utf-8"))
    real_len = len(raw)
    padded = raw + [PAD_BYTE] * (MAX_LEN + 1 - real_len)
    ids = torch.tensor([padded], device=DEVICE)
    inp, target = ids[:, :-1], ids[:, 1:]
    weights = torch.ones(1, MAX_LEN, device=DEVICE)
    weights[:, color_start - 1:real_len - 1] = 20.0
    weights[:, real_len - 1:] = 0.0
    return inp, target, weights


def pc_alm_step(byte_ids, target_ids, loss_weights, track_grad_norms=False):
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
            inp = hs[i].detach().requires_grad_(True)
            pred = bricks[i](inp)
            preds[i + 1] = pred
            rs[i + 1] = (hs[i + 1] - pred).clamp(-R_CLAMP, R_CLAMP)
        last_inp = hs[-1].detach().requires_grad_(True)
        logits = head(last_inp)
        per_tok = F.cross_entropy(logits.reshape(-1, 256), target_ids.reshape(-1),
                                   reduction="none").view(1, -1)
        sup_loss = (per_tok * loss_weights).sum() / loss_weights.sum()
        (grad_last,) = torch.autograd.grad(sup_loss, last_inp)
        grad_last = grad_last.clamp(-R_CLAMP, R_CLAMP)

        new_hs = [hs[0]]
        for i in range(1, N_BRICKS + 1):
            inp_i_minus_1 = hs[i - 1].detach().requires_grad_(True)
            pred_i = bricks[i - 1](inp_i_minus_1)
            r_i = (hs[i] - pred_i).detach().clamp(-R_CLAMP, R_CLAMP)
            if i < N_BRICKS:
                inp_next = hs[i].detach().requires_grad_(True)
                pred_next = bricks[i](inp_next)
                (vjp_term,) = torch.autograd.grad(pred_next, inp_next, grad_outputs=rs[i + 1])
                vjp_term = vjp_term.clamp(-R_CLAMP, R_CLAMP)
            else:
                vjp_term = grad_last
            h_i_new = hs[i] - ETA_H * (r_i - vjp_term + lambdas[i - 1])
            h_i_new = h_i_new.clamp(-H_CLAMP, H_CLAMP)
            new_hs.append(h_i_new.detach())
            lambdas[i - 1] = lambdas[i - 1] + ALPHA * r_i
        hs = new_hs

    grad_norms = []
    for i in range(N_BRICKS):
        inp = hs[i].detach().requires_grad_(True)
        pred = bricks[i](inp)
        r_i = (hs[i + 1] - pred).detach().clamp(-R_CLAMP, R_CLAMP)
        loss_i = (pred * r_i).sum()
        grads = torch.autograd.grad(loss_i, list(bricks[i].parameters()), retain_graph=False,
                                     allow_unused=True)
        with torch.no_grad():
            total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None))
            grad_norms.append(total_norm.item())
            clip_coef = (GRAD_CLIP_NORM / (total_norm + 1e-6)).clamp(max=1.0)
            for p, g in zip(bricks[i].parameters(), grads):
                if g is not None:
                    p += ETA_THETA * g * clip_coef

    io_opt.zero_grad()
    logits = head(hs[-1])
    per_tok = F.cross_entropy(logits.reshape(-1, 256), target_ids.reshape(-1),
                               reduction="none").view(1, -1)
    final_loss = (per_tok * loss_weights).sum() / loss_weights.sum()
    final_loss.backward()
    io_opt.step()
    if track_grad_norms:
        return final_loss.item(), grad_norms
    return final_loss.item()


print("Brick param norms BEFORE any training:", brick_param_norms(), flush=True)

rng = random.Random(0)
for step in range(30):
    color = rng.choice(COLORS)
    text, prefix_len = make_example(rng, color)
    inp, target, loss_weights = pad_to_fixed(text, prefix_len)
    loss, grad_norms = pc_alm_step(inp, target, loss_weights, track_grad_norms=True)
    if step % 5 == 0:
        print(f"step {step}: loss={loss:.4f}  per-brick weight-update GRAD NORMS={[f'{g:.6f}' for g in grad_norms]}  "
              f"param norms={[f'{n:.4f}' for n in brick_param_norms()]}", flush=True)
