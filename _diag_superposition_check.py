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
N_FILLER_TURNS = 8

COLORS = ["blue", "red", "green", "yellow", "purple", "orange", "black", "white", "pink", "gray"]
FILLERS = [
    "The weather is nice today.", "I had cereal for breakfast.", "Traffic was busy this morning.",
    "My neighbor got a new car.", "The meeting ran long.", "I need to buy groceries later.",
    "The movie was pretty good.", "It rained a bit yesterday.", "The coffee shop was crowded.",
    "I finished reading a book.", "The dog needs a walk.", "My phone battery is low.",
]


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


def make_example(rng):
    color = rng.choice(COLORS)
    filler = " ".join(rng.sample(FILLERS, N_FILLER_TURNS))
    prefix = f"My favorite color is {color}. {filler} What is my favorite color? "
    return prefix + color, len(prefix.encode("utf-8"))


model = ByteLM()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
rng = random.Random(0)
print("Retraining the original v1 (sparse/unrelated-filler) setup to reproduce the "
      "orange/green confusion, then checking REPRESENTATION similarity directly...", flush=True)
for step in range(400):
    text, prefix_len = make_example(rng)
    ids = torch.tensor([list(text.encode("utf-8"))], device=DEVICE)
    inp, target = ids[:, :-1], ids[:, 1:]
    opt.zero_grad()
    logits = model(inp)
    loss = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1))
    loss.backward()
    opt.step()

# For each color, get the model's hidden representation (output of the Mamba3 brick, right
# after processing that color WORD in isolation) -- the actual internal representation space,
# not just the byte embedding table.
print("\nPairwise cosine similarity between colors' internal hidden representations:", flush=True)
reps = {}
with torch.no_grad():
    for color in COLORS:
        ids = torch.tensor([list(color.encode("utf-8"))], device=DEVICE)
        x = model.emb(ids)
        h = model.brick.quantized_forward_with_grad(x)
        reps[color] = h[0, -1]  # final hidden state after reading the whole color word

import itertools
sims = []
for a, b in itertools.combinations(COLORS, 2):
    sim = F.cosine_similarity(reps[a].unsqueeze(0), reps[b].unsqueeze(0)).item()
    sims.append((sim, a, b))
sims.sort(reverse=True)

print("\nTop 10 most similar color pairs (most superposed representations):")
for sim, a, b in sims[:10]:
    flag = "  <-- orange/green" if {a, b} == {"orange", "green"} else ""
    print(f"  {a:8s} <-> {b:8s}  cosine_sim={sim:.4f}{flag}")

orange_green_sim = next(s for s, a, b in sims if {a, b} == {"orange", "green"})
rank = next(i for i, (s, a, b) in enumerate(sims) if {a, b} == {"orange", "green"}) + 1
print(f"\norange/green similarity rank: {rank} out of {len(sims)} pairs "
      f"(rank 1 = most similar/most superposed)")
