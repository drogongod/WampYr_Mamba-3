#!/usr/bin/env python3
"""v3, 2026-10-15: Jonathan's refinement on top of v2 -- "English breakfast with English fish
and English sausages and English toad in the hole." Pair the word "favorite" and the target
color DIRECTLY together as often as possible, not just mention the color densely in loosely-
related sentences. v2 repeated "red" a lot (blood, roses, lips) but didn't always pair it
tightly with the literal word "favorite" -- this tests whether that specific direct co-occurrence
is the missing piece, via pure repetitive association, before adding any new weighted mechanism.
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

# Every template pairs "favorite" and {c} DIRECTLY, not just nearby -- maximizing direct
# co-occurrence of the two words that actually matter for answering the question correctly.
TEMPLATES = [
    "My favorite color is {c}.",
    "{C} is my favorite color.",
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
print("Training on direct favorite+color co-occurrence (repetitive association, "
      "the 'English fish, English sausages' shape)...", flush=True)
for step in range(N_TRAIN_STEPS):
    color = rng.choice(COLORS)
    text, prefix_len = make_example(rng, color)
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

# Preserve this trained model -- 2026-10-15, explicit instruction: "keep a copy of the
# training that has worked so well" before building the Forward-Forward comparison.
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "data", "checkpoints",
                                "active_state_v3_cooccurrence_100pct.pt")
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
torch.save({
    "emb": model.emb.state_dict(),
    "mamba": model.brick.mamba.state_dict(),
    "head": model.head.state_dict(),
    "d_model": D_MODEL,
    "note": "v3 direct favorite+color co-occurrence training, 100% held-out, "
            "generalized to Jonathan's own out-of-template example. 2026-10-15.",
}, CHECKPOINT_PATH)
print(f"\n[checkpoint saved] {CHECKPOINT_PATH}", flush=True)

print("\n=== JONATHAN'S EXACT EXAMPLE (red, vampire/blood imagery, NOT in this template family) ===",
      flush=True)
gen = model.generate(JONATHANS_EXAMPLE, n=6)
answer = gen[len(JONATHANS_EXAMPLE):]
print(f"  generated answer: {answer!r}  (target: 'red')  {'OK' if answer.strip().startswith('red') else 'WRONG'}",
      flush=True)

print("\n=== HELD-OUT EVALUATION (fresh random examples, all 10 colors) ===", flush=True)
eval_rng = random.Random(999)
correct = 0
orange_results = []
n_eval = 30
for _ in range(n_eval):
    color = eval_rng.choice(COLORS)
    text, prefix_len = make_example(eval_rng, color)
    prompt = text[:prefix_len]
    gen = model.generate(prompt, n=len(max(COLORS, key=len)))
    answer = gen[len(prompt):]
    got_it = answer.strip().startswith(color)
    correct += int(got_it)
    if color == "orange":
        orange_results.append((answer, got_it))
    print(f"  fact={color:8s} -> generated={answer[:12]!r}  {'OK' if got_it else 'WRONG'}", flush=True)

print(f"\n=== RESULT ===")
print(f"Direct co-occurrence recall accuracy: {correct}/{n_eval} = {100*correct/n_eval:.1f}%")
print(f"orange-specific results (previously confused with green both times): {orange_results}")
