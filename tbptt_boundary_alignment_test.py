#!/usr/bin/env python3
"""TBPTT BOUNDARY-ALIGNMENT CONTROL, 2026-10-16 -- a reviewing model (OpenAI) caught a real
confound in tbptt_window_sweep.py: chunk boundaries were anchored to the START of the sequence
(byte 0, WINDOW, 2*WINDOW, ...), so the LAST chunk (the only one whose gradient can possibly
reach the answer) had a size determined by `sequence_length % WINDOW` -- pure accident of
sequence length, nothing to do with the real credit-horizon question. Verified directly: in 4 of
5 sampled examples, window=64's last chunk (15-39 bytes) ended up ENTIRELY AFTER the last color
restatement, so that condition never had a fair chance regardless of whether 64 bytes would
otherwise be enough.

THE FIX: anchor chunk boundaries from the END of the sequence instead. Any short leftover chunk
goes at the FRONT (where it doesn't matter -- the front is far from the answer either way); every
window counted backward from the answer, especially the LAST one, is then exactly WINDOW bytes,
with zero accident. This directly tests "is a guaranteed W bytes of credit immediately before
the query enough" -- the real question -- separated from boundary luck.

Runs window={64, 256} under BOTH anchor modes for a direct, controlled comparison, same task,
same model, same training budget (1200 examples, matching the real sweep) as the original.
"""
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"D:\mamba3_clean_ref")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba3 import Mamba3 as CleanMamba3

WINDOW = int(sys.argv[1])
ANCHOR = sys.argv[2]  # "start" or "end"
assert ANCHOR in ("start", "end")
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

torch.manual_seed(SEED)
random.seed(SEED)
DEVICE = "cuda"
D_MODEL = 128

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


def boundary_set(n_steps: int, window: int, anchor: str) -> set:
    """Returns the set of t-indices (0-indexed step count) at which a window CLOSES (i.e., after
    processing step t, detach). start-anchor: windows are [0,W),[W,2W),... (last one short).
    end-anchor: windows counted backward from n_steps, so the LAST window is exactly W, any
    short leftover is at the FRONT."""
    closes = set()
    if anchor == "start":
        count = 0
        for t in range(n_steps):
            count += 1
            if count >= window:
                closes.add(t)
                count = 0
    else:  # end-anchor
        remainder = n_steps % window
        first_chunk = remainder if remainder != 0 else window
        # first chunk closes at index (first_chunk - 1)
        closes.add(first_chunk - 1)
        t = first_chunk - 1
        while t + window < n_steps:
            t += window
            closes.add(t)
    return closes


class RollingWindowModel(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(256, d_model).to(DEVICE)
        self.mamba = CleanMamba3(d_model=d_model, d_state=32, expand=2, headdim=16, ngroups=1,
                                   is_mimo=False).to(DEVICE)
        self.head = nn.Linear(d_model, 256).to(DEVICE)

    def train_sequence(self, byte_ids, opt):
        n_steps = len(byte_ids) - 1
        closes = boundary_set(n_steps, WINDOW, ANCHOR)
        angle_state, ssm_state, bx_prev = self.mamba.allocate_inference_cache(1, device=DEVICE)
        total_loss, n_losses = 0.0, 0
        window_loss = 0.0
        for t in range(n_steps):
            inp_byte = torch.tensor([byte_ids[t]], device=DEVICE)
            target_byte = torch.tensor([byte_ids[t + 1]], device=DEVICE)
            x = self.emb(inp_byte)
            out, angle_state, ssm_state, bx_prev = self.mamba.step(x, angle_state, ssm_state, bx_prev)
            logits = self.head(out)
            loss = F.cross_entropy(logits, target_byte)
            window_loss = window_loss + loss
            total_loss += loss.item()
            n_losses += 1
            if t in closes:
                opt.zero_grad()
                window_loss.backward()
                opt.step()
                angle_state, ssm_state, bx_prev = angle_state.detach(), ssm_state.detach(), bx_prev.detach()
                window_loss = 0.0
        if window_loss != 0.0 and isinstance(window_loss, torch.Tensor) and window_loss.requires_grad:
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
rng = random.Random(SEED)

N_TRAIN_EXAMPLES = 1200
print(f"=== WINDOW={WINDOW} ANCHOR={ANCHOR} SEED={SEED} ===", flush=True)
for i in range(N_TRAIN_EXAMPLES):
    color = rng.choice(COLORS)
    text, _ = make_example(rng, color)
    byte_ids = list(text.encode("utf-8"))
    avg_loss = model.train_sequence(byte_ids, opt)
    if i % 100 == 0:
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

print(f"\n=== RESULT window={WINDOW} anchor={ANCHOR} seed={SEED} ===")
print(f"accuracy: {correct}/{n_eval} = {100*correct/n_eval:.1f}%  jonathans_example: {jon_pred} ({'OK' if jon_ok else 'WRONG'})")
