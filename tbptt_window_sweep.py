#!/usr/bin/env python3
"""TBPTT WINDOW SWEEP, 2026-10-16 -- the single experiment multiple independent model reviews
converged on as the cheapest, most direct test of the "credit-horizon" theory for why TBPTT(3)
scored 3.3% while full backprop scores 100%: sweep the truncation window with everything else
identical (same task, same model size, same training budget) and see whether accuracy recovers
gradually as the window grows (confirming a credit-horizon effect, with a knee near the real
evidence-to-query distance) or stays flat until the window reaches the full sequence (pointing
at something else entirely).

Forward-equivalence between chunked-detach and uninterrupted processing was verified directly
beforehand (_diag_tbptt_forward_equivalence.py, max diff 1.1e-6) -- chunking itself does not
corrupt the computation, so any accuracy difference here is attributable to the gradient
truncation itself, not a confound in how windowing was implemented.

Usage: python tbptt_window_sweep.py <window_size_or_"full">
"""
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"D:\mamba3_clean_ref")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba3 import Mamba3 as CleanMamba3

WINDOW_ARG = sys.argv[1] if len(sys.argv) > 1 else "3"

torch.manual_seed(0)
random.seed(0)
DEVICE = "cuda"
D_MODEL = 128
WINDOW = 10**9 if WINDOW_ARG == "full" else int(WINDOW_ARG)  # effectively "never truncate"

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
JONATHANS_EXAMPLE = (
    "I am a vampire so my favorite thing is red human blood. I am a vampire vampires drink red "
    "blood. I love the red lips that kiss the red roses the cover of the grave. my favorite site "
    "is red blood that sparkles on a woman's Red lips as they kiss, Red roses are my most loved "
    "flower. Strawberry ice cream would be my favorite if I could eat as red berries are my "
    "favorite berries. My favorite flowers are red Roses.. What is my favorite color? "
)


def make_example(rng, color):
    n_sentences = rng.randint(5, 8)
    sentences = rng.sample(TEMPLATES, min(n_sentences, len(TEMPLATES)))
    body = " ".join(t.format(c=color, C=color.capitalize()) for t in sentences)
    prefix = f"{body} What is my favorite color? "
    return prefix + color, len(prefix.encode("utf-8"))


class RollingWindowModel(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(256, d_model).to(DEVICE)
        self.mamba = CleanMamba3(d_model=d_model, d_state=32, expand=2, headdim=16, ngroups=1,
                                   is_mimo=False).to(DEVICE)
        self.head = nn.Linear(d_model, 256).to(DEVICE)

    def train_sequence(self, byte_ids, opt):
        angle_state, ssm_state, bx_prev = self.mamba.allocate_inference_cache(1, device=DEVICE)
        total_loss, n_losses = 0.0, 0
        window_loss, window_count = 0.0, 0
        for t in range(len(byte_ids) - 1):
            inp_byte = torch.tensor([byte_ids[t]], device=DEVICE)
            target_byte = torch.tensor([byte_ids[t + 1]], device=DEVICE)
            x = self.emb(inp_byte)
            out, angle_state, ssm_state, bx_prev = self.mamba.step(x, angle_state, ssm_state, bx_prev)
            logits = self.head(out)
            loss = F.cross_entropy(logits, target_byte)
            window_loss = window_loss + loss
            window_count += 1
            total_loss += loss.item()
            n_losses += 1
            if window_count >= WINDOW:
                opt.zero_grad()
                window_loss.backward()
                opt.step()
                angle_state, ssm_state, bx_prev = angle_state.detach(), ssm_state.detach(), bx_prev.detach()
                window_loss, window_count = 0.0, 0
        if window_count > 0:
            opt.zero_grad()
            window_loss.backward()
            opt.step()
        return total_loss / max(n_losses, 1)

    @torch.no_grad()
    def score_completion(self, prompt, candidate):
        full = prompt + candidate
        byte_ids = list(full.encode("utf-8"))
        prompt_len = len(prompt.encode("utf-8"))
        angle_state, ssm_state, bx_prev = self.mamba.allocate_inference_cache(1, device=DEVICE)
        total_nll, n = 0.0, 0
        for t in range(len(byte_ids) - 1):
            inp_byte = torch.tensor([byte_ids[t]], device=DEVICE)
            target_byte = torch.tensor([byte_ids[t + 1]], device=DEVICE)
            x = self.emb(inp_byte)
            out, angle_state, ssm_state, bx_prev = self.mamba.step(x, angle_state, ssm_state, bx_prev)
            logits = self.head(out)
            if t >= prompt_len - 1:
                total_nll += F.cross_entropy(logits, target_byte).item()
                n += 1
        return -total_nll / max(n, 1)


model = RollingWindowModel()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
rng = random.Random(0)

N_TRAIN_EXAMPLES = 1200
print(f"=== WINDOW={WINDOW_ARG} ===", flush=True)
print(f"Training (step-by-step, truncation window={WINDOW_ARG})...", flush=True)
for i in range(N_TRAIN_EXAMPLES):
    color = rng.choice(COLORS)
    text, _ = make_example(rng, color)
    byte_ids = list(text.encode("utf-8"))
    avg_loss = model.train_sequence(byte_ids, opt)
    if i % 30 == 0:
        print(f"  example {i}: avg_loss={avg_loss:.4f}", flush=True)

scores = {c: model.score_completion(JONATHANS_EXAMPLE, c) for c in COLORS}
jon_pred = max(scores, key=scores.get)
jon_ok = jon_pred == "red"

eval_rng = random.Random(999)
correct = 0
n_eval = 30
for _ in range(n_eval):
    color = eval_rng.choice(COLORS)
    text, prefix_len = make_example(eval_rng, color)
    prompt = text[:prefix_len]
    scores = {c: model.score_completion(prompt, c) for c in COLORS}
    pred = max(scores, key=scores.get)
    correct += int(pred == color)

print(f"\n=== RESULT window={WINDOW_ARG} ===")
print(f"accuracy: {correct}/{n_eval} = {100*correct/n_eval:.1f}%  jonathans_example: {jon_pred} ({'OK' if jon_ok else 'WRONG'})")
