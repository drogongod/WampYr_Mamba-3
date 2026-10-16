#!/usr/bin/env python3
"""THE REAL SISTER-SPLIT MECHANISM, faithfully ported from sovereign_core.py's _fire_relays(),
2026-10-15 -- not gradient truncation (that was a mismatched translation of a non-gradient
system). The actual mechanism: when a brick's prediction_error exceeds a threshold, the training
signal SPLITS -- the home brick still gets (1 - split_ratio) of normal training on this example,
and split_ratio's worth goes SECONDHAND to a SISTER brick (never the source, matching the real
code's "never the source itself"). Real code's own fallback when no prototype exists yet:
random sister selection -- used here directly, not a simplification of convenience.

A/B test: same 5-brick cluster, same data, same seeds -- WITH splitting vs WITHOUT (ablation),
on the proven favorite-color direct-co-occurrence recall task (100% full-backprop baseline).
"""
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba3_quinary_brick import Mamba3QuinaryBrick

DEVICE = "cuda"
D_MODEL = 64
N_BRICKS = 5
SPLIT_THRESHOLD = 3.0   # real code's own init range was 1.5-4.0
SPLIT_RATIO = 0.5       # real code's own init range was 0.4-0.6

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
# Fixed home-brick assignment, round-robin -- stand-in for the cluster's real content routing.
HOME_BRICK = {c: i % N_BRICKS for i, c in enumerate(COLORS)}


def make_example(rng, color):
    n_sentences = rng.randint(5, 8)
    sentences = rng.sample(TEMPLATES, min(n_sentences, len(TEMPLATES)))
    body = " ".join(t.format(c=color, C=color.capitalize()) for t in sentences)
    prefix = f"{body} What is my favorite color? "
    return prefix + color, len(prefix.encode("utf-8"))


class BrickLM(nn.Module):
    def __init__(self, seed, d_model=D_MODEL):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(256, d_model).to(DEVICE)
        self.brick = Mamba3QuinaryBrick(d_model=d_model, d_state=24, headdim=16, device=DEVICE)
        self.head = nn.Linear(d_model, 256).to(DEVICE)
        self.opt = torch.optim.Adam(self.parameters(), lr=3e-3)

    def loss_on(self, text: str, color_start: int, lr_scale: float = 1.0) -> float:
        ids = torch.tensor([list(text.encode("utf-8"))], device=DEVICE)
        inp, target = ids[:, :-1], ids[:, 1:]
        x = self.emb(inp)
        h = self.brick.quantized_forward_with_grad(x)
        logits = self.head(h)
        per_tok = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1), reduction="none").view(1, -1)
        weights = torch.ones_like(per_tok)
        weights[:, color_start - 1:] = 20.0
        loss = (per_tok * weights).sum() / weights.sum()
        for g in self.opt.param_groups:
            g["lr"] = 3e-3 * lr_scale
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def generate(self, prompt: str, n: int) -> str:
        ids = list(prompt.encode("utf-8"))
        for _ in range(n):
            x = torch.tensor([ids], device=DEVICE)
            emb = self.emb(x)
            h = self.brick.quantized_forward_with_grad(emb)
            logits = self.head(h)
            nxt = int(torch.argmax(logits[0, -1]).item())
            ids.append(nxt)
        return bytes(ids[len(prompt.encode("utf-8")):]).decode("utf-8", errors="replace")


def run_condition(use_split: bool, seed: int = 0) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    bricks = [BrickLM(seed=seed + 1000 * i) for i in range(N_BRICKS)]
    rng = random.Random(seed)
    split_events = 0

    N_STEPS = 800
    for step in range(N_STEPS):
        color = rng.choice(COLORS)
        text, prefix_len = make_example(rng, color)
        home_idx = HOME_BRICK[color]
        home = bricks[home_idx]

        if use_split:
            # probe the error WITHOUT taking a full step first would need a no-grad forward;
            # simplest faithful read: take the home step at reduced/full scale based on last
            # known error for this brick, tracked across steps (real code reads b.prediction_error
            # from the PREVIOUS cycle too, not a fresh lookahead).
            lr_scale = 1.0
            if hasattr(home, "_last_err") and home._last_err > SPLIT_THRESHOLD:
                lr_scale = 1.0 - SPLIT_RATIO
            loss = home.loss_on(text, prefix_len, lr_scale=lr_scale)
            home._last_err = loss
            if loss > SPLIT_THRESHOLD:
                split_events += 1
                # SISTER: never the source, matching real code exactly. No prototype tracked in
                # this test, so random sister selection -- the real code's own documented
                # fallback for this exact situation, not a simplification of convenience.
                sister_idx = rng.choice([i for i in range(N_BRICKS) if i != home_idx])
                bricks[sister_idx].loss_on(text, prefix_len, lr_scale=SPLIT_RATIO)
        else:
            loss = home.loss_on(text, prefix_len, lr_scale=1.0)
            home._last_err = loss

    print(f"  [{'WITH' if use_split else 'WITHOUT'} split] split_events={split_events}/{N_STEPS}",
          flush=True)

    # Evaluation: ask each brick for its own colors (home-brick routing at inference, matching
    # how the real cluster would route by content).
    eval_rng = random.Random(999)
    correct = 0
    n_eval = 30
    results = []
    for _ in range(n_eval):
        color = eval_rng.choice(COLORS)
        text, prefix_len = make_example(eval_rng, color)
        prompt = text[:prefix_len]
        home = bricks[HOME_BRICK[color]]
        answer = home.generate(prompt, n=len(max(COLORS, key=len)))
        got_it = answer.strip().startswith(color)
        correct += int(got_it)
        results.append((color, answer, got_it))

    jon_home = bricks[HOME_BRICK["red"]]
    jon_answer = jon_home.generate(JONATHANS_EXAMPLE, n=6)
    jon_ok = jon_answer.strip().startswith("red")

    return {"accuracy": correct / n_eval, "n_correct": correct, "n_eval": n_eval,
            "jonathans_example_ok": jon_ok, "jonathans_answer": jon_answer}, \
           {"results": results}


print("=== CONDITION A: WITHOUT sister-split (baseline ablation) ===", flush=True)
res_without, detail_without = run_condition(use_split=False, seed=0)
print(f"  accuracy: {res_without['n_correct']}/{res_without['n_eval']} = "
      f"{100*res_without['accuracy']:.1f}%  Jonathan's example: {res_without['jonathans_example_ok']}",
      flush=True)

print("\n=== CONDITION B: WITH sister-split (the real mechanism) ===", flush=True)
res_with, detail_with = run_condition(use_split=True, seed=0)
print(f"  accuracy: {res_with['n_correct']}/{res_with['n_eval']} = "
      f"{100*res_with['accuracy']:.1f}%  Jonathan's example: {res_with['jonathans_example_ok']}",
      flush=True)

print("\n=== RESULT ===")
print(f"WITHOUT split: {100*res_without['accuracy']:.1f}%")
print(f"WITH split:    {100*res_with['accuracy']:.1f}%")
print(f"(Full single-brick backprop baseline, v3: 100%. This test uses a {N_BRICKS}-brick "
      f"cluster with fixed content routing, not single-brick, for a fair A/B on the split "
      f"mechanism specifically.)")
