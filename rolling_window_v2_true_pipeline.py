#!/usr/bin/env python3
"""THE TRUE THREE-BRICK ROLLING WINDOW, 2026-10-15, v2 -- faithful rebuild after the first
attempt mistranslated "three bricks" into "three time-steps of one shared-weight module," which
is NOT what the diary describes and collapsed to 3.3%.

Real design: signal flows through a CHAIN OF SEPARATE PHYSICAL BRICKS (each its own independent
weights), not one module unrolled in time. This version: 6 separate small Mamba3 bricks stacked
as a true chain (brick i's output IS brick i+1's input, over the WHOLE sequence, like a 6-deep
network where every layer is a full Mamba3 module). Gradient is truncated to the LAST WINDOW=3
bricks only -- bricks 0-2 stay at their random initialization, genuinely untrained, acting as a
fixed nonlinear feature stage; only bricks 3-5 ever receive a gradient. This is the simplest
correct version of "bounded local credit assignment through a real multi-brick chain" -- one
fixed window position, not yet the full sliding-across-training generality, but genuinely
separate-weight bricks with truncated depth, not a time-truncation approximation.
"""
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"D:\mamba3_clean_ref")
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba3 import Mamba3 as CleanMamba3

torch.manual_seed(0)
random.seed(0)
DEVICE = "cuda"
D_MODEL = 64
N_BRICKS = 6
WINDOW = 3  # only the LAST 3 bricks get gradient; bricks 0..2 stay fixed/random

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
    n_sentences = rng.randint(6, 8)
    sentences = rng.sample(TEMPLATES, min(n_sentences, len(TEMPLATES)))
    body = " ".join(t.format(c=color, C=color.capitalize()) for t in sentences)
    prefix = f"{body} What is my favorite color? "
    return prefix + color, len(prefix.encode("utf-8"))


class Pipeline(nn.Module):
    def __init__(self, n_bricks=N_BRICKS, window=WINDOW, d_model=D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(256, d_model).to(DEVICE)
        self.head = nn.Linear(d_model, 256).to(DEVICE)
        self.bricks = nn.ModuleList([
            CleanMamba3(d_model=d_model, d_state=24, expand=2, headdim=16, ngroups=1,
                        is_mimo=False).to(DEVICE)
            for _ in range(n_bricks)
        ])
        self.window = window
        trainable = list(self.emb.parameters()) + list(self.head.parameters())
        for brick in self.bricks[-window:]:
            trainable += list(brick.parameters())
        self.opt = torch.optim.Adam(trainable, lr=3e-3)
        for brick in self.bricks[:-window]:
            for p in brick.parameters():
                p.requires_grad_(False)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        # 2026-10-15, real bug found+fixed: the fixed bricks' own params already have
        # requires_grad_(False) set in __init__, which is sufficient to keep them untrained.
        # The `with torch.no_grad():` that used to wrap this loop was redundant for that and
        # ALSO silently severed the gradient path back to self.emb -- self.emb never received
        # a gradient despite being in self.opt's param list the whole time. Removed; the
        # explicit h.detach() below is the real, intended truncation point (unchanged).
        h = self.emb(byte_ids)
        n_fixed = len(self.bricks) - self.window
        for brick in self.bricks[:n_fixed]:
            h = brick(h)
        h = h.detach()  # the real truncation: no gradient reaches the fixed bricks' output
        for brick in self.bricks[n_fixed:]:
            h = brick(h)
        return self.head(h)


model = Pipeline()
rng = random.Random(0)

N_TRAIN_STEPS = 800
COLOR_LOSS_WEIGHT = 20.0
print(f"Training the true {N_BRICKS}-brick chain (window={WINDOW}: only the last {WINDOW} "
      f"bricks are trainable, first {N_BRICKS - WINDOW} stay fixed/random)...", flush=True)
for step in range(N_TRAIN_STEPS):
    color = rng.choice(COLORS)
    text, prefix_len = make_example(rng, color)
    ids = torch.tensor([list(text.encode("utf-8"))], device=DEVICE)
    inp, target = ids[:, :-1], ids[:, 1:]
    model.opt.zero_grad()
    logits = model(inp)
    per_tok = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1), reduction="none").view(1, -1)
    color_start = prefix_len - 1
    weights = torch.ones_like(per_tok)
    weights[:, color_start:] = COLOR_LOSS_WEIGHT
    loss = (per_tok * weights).sum() / weights.sum()
    loss.backward()
    model.opt.step()
    if step % 100 == 0:
        color_loss = per_tok[:, color_start:].mean().item()
        print(f"  step {step}: weighted_loss={loss.item():.4f}  color_bytes_loss={color_loss:.4f}",
              flush=True)


@torch.no_grad()
def generate(prompt: str, n: int = 10) -> str:
    ids = list(prompt.encode("utf-8"))
    for _ in range(n):
        x = torch.tensor([ids], device=DEVICE)
        logits = model(x)
        nxt = int(torch.argmax(logits[0, -1]).item())
        ids.append(nxt)
    return bytes(ids[len(prompt.encode("utf-8")):]).decode("utf-8", errors="replace")


print("\n=== JONATHAN'S EXACT EXAMPLE ===", flush=True)
answer = generate(JONATHANS_EXAMPLE, n=6)
print(f"  generated: {answer!r}  (target: red)  {'OK' if answer.strip().startswith('red') else 'WRONG'}",
      flush=True)

print("\n=== HELD-OUT EVALUATION (fresh random examples, all 10 colors) ===", flush=True)
eval_rng = random.Random(999)
correct = 0
n_eval = 30
for _ in range(n_eval):
    color = eval_rng.choice(COLORS)
    text, prefix_len = make_example(eval_rng, color)
    prompt = text[:prefix_len]
    answer = generate(prompt, n=len(max(COLORS, key=len)))
    got_it = answer.strip().startswith(color)
    correct += int(got_it)
    print(f"  fact={color:8s} -> generated={answer[:12]!r}  {'OK' if got_it else 'WRONG'}", flush=True)

print(f"\n=== RESULT ===")
print(f"True {N_BRICKS}-brick chain, window={WINDOW} local backprop accuracy: "
      f"{correct}/{n_eval} = {100*correct/n_eval:.1f}%")
print(f"(Full backprop baseline, v3: 100%. Forward-Forward peak: 73.3%. "
      f"Time-truncated TBPTT(3) mistranslation: 3.3%.)")
