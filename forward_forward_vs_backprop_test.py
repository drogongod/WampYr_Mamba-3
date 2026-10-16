#!/usr/bin/env python3
"""Forward-Forward (Hinton, 2022, 'The Forward-Forward Algorithm: Some Preliminary
Investigations') vs. standard backprop, same favorite-color recall task v3 already solved at
100% with full backprop (see active_state_v3_cooccurrence_100pct.pt, preserved separately).

HONEST SCOPE, stated plainly, not oversold: Mamba3 is a standard PyTorch nn.Module, so the actual
gradient computation under the hood still runs through torch.autograd. What's actually being
tested is Hinton's real, published LOCAL, CONTRASTIVE, GOODNESS-BASED objective -- no cross-
entropy next-token loss, no explicit generation target, the decision is made by comparing a
scalar "goodness" across candidate completions -- not a literal synapse-level, no-autograd rule.
This is the real, meaningfully-different-from-backprop PART of Forward-Forward (no global loss
signal computed from a correct/incorrect final answer and propagated backward through time via
BPTT); it is not the full biological end of the local-learning spectrum.

Goodness = sum of squared hidden activations over the sequence (Hinton's own definition).
Positive example: context + the TRUE color appended. Negative example: context + a WRONG color.
Train: push positive goodness above a threshold, negative goodness below it (logistic loss on
goodness - threshold, Hinton's own formulation) -- no label/target token predicted anywhere.
Inference: try each of the 10 candidate colors appended, pick whichever gives highest goodness.
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
THRESHOLD = 2.0  # Hinton's own convention: goodness measured relative to a threshold

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
JONATHANS_EXAMPLE_PREFIX = (
    "I am a vampire so my favorite thing is red human blood. I am a vampire vampires drink red "
    "blood. I love the red lips that kiss the red roses the cover of the grave. my favorite site "
    "is red blood that sparkles on a woman's Red lips as they kiss, Red roses are my most loved "
    "flower. Strawberry ice cream would be my favorite if I could eat as red berries are my "
    "favorite berries. My favorite flowers are red Roses.. What is my favorite color? "
)


def make_context(rng, color):
    n_sentences = rng.randint(5, 8)
    sentences = rng.sample(TEMPLATES, min(n_sentences, len(TEMPLATES)))
    body = " ".join(t.format(c=color, C=color.capitalize()) for t in sentences)
    return f"{body} What is my favorite color? "


class GoodnessModel(nn.Module):
    """No output head at all -- Forward-Forward needs no generation target, only a scalar
    goodness over hidden activations."""
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(256, d_model).to(DEVICE)
        self.brick = Mamba3QuinaryBrick(d_model=d_model, d_state=32, headdim=16, device=DEVICE)

    def goodness(self, text: str, answer_bytes: int) -> torch.Tensor:
        """2026-10-15, real fix: summing over the WHOLE sequence let shared context (identical
        between positive/negative) dilute the tiny fraction of signal that actually
        distinguishes them -- same dilution bug already found twice today in the cross-entropy
        formulation, now fixed the same way: restrict goodness to the region that actually
        differs (the appended color word), not the whole sequence."""
        ids = torch.tensor([list(text.encode("utf-8"))], device=DEVICE)
        x = self.emb(ids)
        h = self.brick.quantized_forward_with_grad(x)
        return (h[:, -answer_bytes:] ** 2).sum()


model = GoodnessModel()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
rng = random.Random(0)

N_TRAIN_STEPS = 8000
print("Training with Forward-Forward's local, contrastive, goodness-based objective "
      "(no cross-entropy, no generation target)...", flush=True)
for step in range(N_TRAIN_STEPS):
    color = rng.choice(COLORS)
    wrong_color = rng.choice([c for c in COLORS if c != color])
    context = make_context(rng, color)

    pos_text = context + color
    neg_text = context + wrong_color

    g_pos = model.goodness(pos_text, len(color.encode("utf-8")))
    g_neg = model.goodness(neg_text, len(wrong_color.encode("utf-8")))

    # 2026-10-15, reverted to Hinton's ACTUAL published formulation -- my own added margin
    # (relu, hinge-shaped: constant gradient then suddenly zero) was a modification I layered on
    # top, not what's documented, and hinge losses are known to interact badly with Adam's
    # adaptive step sizes (sudden gradient discontinuities). Real Forward-Forward: p = sigmoid
    # (goodness - threshold), trained via standard BCE -- smooth everywhere, naturally saturates
    # (vanishing gradient) once confidently correct, no hinge, no margin hack.
    EPS = 1e-6
    p_pos = torch.sigmoid(g_pos - THRESHOLD)
    p_neg = torch.sigmoid(g_neg - THRESHOLD)
    loss = -torch.log(p_pos + EPS) - torch.log(1 - p_neg + EPS)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 500 == 0:
        print(f"  step {step}: loss={loss.item():.4f}  g_pos={g_pos.item():.3f}  "
              f"g_neg={g_neg.item():.3f}", flush=True)

print(f"\nFinal: g_pos={g_pos.item():.3f}  g_neg={g_neg.item():.3f}  "
      f"(want g_pos high, g_neg low, both relative to threshold={THRESHOLD})", flush=True)


@torch.no_grad()
def predict(context_text: str) -> tuple[str, dict]:
    scores = {c: model.goodness(context_text + c, len(c.encode("utf-8"))).item() for c in COLORS}
    best = max(scores, key=scores.get)
    return best, scores


print("\n=== JONATHAN'S EXACT EXAMPLE ===", flush=True)
pred, scores = predict(JONATHANS_EXAMPLE_PREFIX)
print(f"  predicted: {pred}  (target: red)  {'OK' if pred == 'red' else 'WRONG'}", flush=True)
print(f"  all scores: {scores}", flush=True)

print("\n=== HELD-OUT EVALUATION (fresh random examples, all 10 colors) ===", flush=True)
eval_rng = random.Random(999)
correct = 0
n_eval = 30
for _ in range(n_eval):
    color = eval_rng.choice(COLORS)
    context = make_context(eval_rng, color)
    pred, _ = predict(context)
    got_it = pred == color
    correct += int(got_it)
    print(f"  fact={color:8s} -> predicted={pred:8s}  {'OK' if got_it else 'WRONG'}", flush=True)

print(f"\n=== RESULT ===")
print(f"Forward-Forward goodness-based recall accuracy: {correct}/{n_eval} = {100*correct/n_eval:.1f}%")
print(f"(Backprop baseline on this exact task family, v3: 100%, checkpoint preserved at "
      f"data/checkpoints/active_state_v3_cooccurrence_100pct.pt)")
