#!/usr/bin/env python3
"""v2 of the active-state folding test, 2026-10-15: Jonathan's refinement -- instead of a single
early mention buried in UNRELATED filler, test REINFORCED DENSITY: the target color mentioned
repeatedly across many varied, thematically-reinforcing sentences, then asked for at the end.

No host/donor anywhere in this architecture -- a single standalone Mamba3 instance, so whatever
result comes out is host-off by construction, nothing to accidentally fall back to.

Includes Jonathan's own literal example (vampire/blood imagery, color=red) as a direct concrete
case, plus a generalized, parameterized version across all 10 colors for genuine held-out
generalization testing (not just one hardcoded case).
"""
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba3_quinary_brick import Mamba3QuinaryBrick

torch.manual_seed(0)
random.seed(0)
DEVICE = "cuda"
D_MODEL = 128

COLORS = ["blue", "red", "green", "yellow", "purple", "orange", "black", "white", "pink", "gray"]

# Reinforcement sentence templates -- {c} is the target color, repeated densely, varied framing
# each time (Jonathan's own shape: blood/lips/roses/berries/flowers, all pointing to one color).
TEMPLATES = [
    "I am a vampire, so my favorite thing is {c} human blood.",
    "I love the {c} lips that kiss the {c} roses by the cover of the grave.",
    "My favorite sight is {c} blood that sparkles on a lover's {c} lips as they kiss.",
    "{C} roses are my most loved flower.",
    "Strawberry ice cream would be my favorite if I could eat, as {c} berries are my favorite berries.",
    "My favorite flowers are {c} roses.",
    "The {c} moon rises over {c} fields of {c} flowers every night I hunt.",
    "Nothing stirs me like the taste of {c}, the color of {c} silk, the smell of {c} wine.",
]

JONATHANS_EXAMPLE = (
    "I am a vampire so my favorite thing is red human blood. I am a vampire vampires drink red "
    "blood. I love the red lips that kiss the red roses the cover of the grave. my favorite site "
    "is red blood that sparkles on a woman's Red lips as they kiss, Red roses are my most loved "
    "flower. Strawberry ice cream would be my favorite if I could eat as red berries are my "
    "favorite berries. My favorite flowers are red Roses.. What is my favorite color? "
)


def make_density_example(rng, color):
    n_sentences = rng.randint(5, 8)
    sentences = rng.sample(TEMPLATES, min(n_sentences, len(TEMPLATES)))
    body = " ".join(t.format(c=color, C=color.capitalize()) for t in sentences)
    prefix = f"{body} What is my favorite color? "
    text = prefix + color
    return text, len(prefix.encode("utf-8"))


class ByteLM(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(256, d_model).to(DEVICE)
        self.brick = Mamba3QuinaryBrick(d_model=d_model, d_state=32, headdim=16, device=DEVICE)
        self.head = nn.Linear(d_model, 256).to(DEVICE)

    def forward(self, byte_ids):
        x = self.emb(byte_ids)
        h = self.brick.quantized_forward_with_grad(x)
        return self.head(h)

    @torch.no_grad()
    def generate(self, prompt: str, n: int = 10) -> str:
        ids = list(prompt.encode("utf-8"))
        for _ in range(n):
            x = torch.tensor([ids], device=DEVICE)
            logits = self.forward(x)
            next_id = int(torch.argmax(logits[0, -1]).item())
            ids.append(next_id)
        return bytes(ids).decode("utf-8", errors="replace")


model = ByteLM()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
rng = random.Random(0)

N_TRAIN_STEPS = 800
COLOR_LOSS_WEIGHT = 20.0
print(f"Training on REINFORCED-DENSITY recall (target color repeated 5-8x across varied "
      f"sentences, not buried once in unrelated filler)...", flush=True)
for step in range(N_TRAIN_STEPS):
    color = rng.choice(COLORS)
    text, prefix_len = make_density_example(rng, color)
    ids = torch.tensor([list(text.encode("utf-8"))], device=DEVICE)
    inp, target = ids[:, :-1], ids[:, 1:]
    opt.zero_grad()
    logits = model(inp)
    per_tok = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1), reduction="none").view(1, -1)
    color_start = prefix_len - 1
    weights = torch.ones_like(per_tok)
    weights[:, color_start:] = COLOR_LOSS_WEIGHT
    loss = (per_tok * weights).sum() / weights.sum()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        color_loss = per_tok[:, color_start:].mean().item()
        print(f"  step {step}: weighted_loss={loss.item():.4f}  color_bytes_loss={color_loss:.4f}",
              flush=True)

print("\n=== JONATHAN'S EXACT EXAMPLE (red, vampire/blood imagery) ===", flush=True)
gen = model.generate(JONATHANS_EXAMPLE, n=6)
answer = gen[len(JONATHANS_EXAMPLE):]
print(f"  generated answer: {answer!r}  (target: 'red')  {'OK' if answer.strip().startswith('red') else 'WRONG'}",
      flush=True)

print("\n=== HELD-OUT EVALUATION (fresh random examples, all 10 colors) ===", flush=True)
eval_rng = random.Random(999)
correct = 0
n_eval = 30
for _ in range(n_eval):
    color = eval_rng.choice(COLORS)
    text, prefix_len = make_density_example(eval_rng, color)
    prompt = text[:prefix_len]
    gen = model.generate(prompt, n=len(max(COLORS, key=len)))
    answer = gen[len(prompt):]
    got_it = answer.strip().startswith(color)
    correct += int(got_it)
    print(f"  fact={color:8s} -> generated={answer[:12]!r}  {'OK' if got_it else 'WRONG'}", flush=True)

print(f"\n=== RESULT ===")
print(f"Reinforced-density recall accuracy: {correct}/{n_eval} = {100*correct/n_eval:.1f}%")
print("No host/donor anywhere in this architecture -- host-off by construction.")
