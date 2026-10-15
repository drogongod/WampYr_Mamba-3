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


model = ByteLM()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
rng = random.Random(0)

def make_example(rng):
    color = rng.choice(COLORS)
    filler = " ".join(rng.sample(FILLERS, N_FILLER_TURNS))
    prefix = f"My favorite color is {color}. {filler} What is my favorite color? "
    text = prefix + color
    return text, len(prefix.encode("utf-8")), color

print("Training 400 steps, tracking loss AT THE COLOR-ANSWER BYTES specifically "
      "vs the rest of the sequence...", flush=True)
for step in range(400):
    text, prefix_len, color = make_example(rng)
    ids = torch.tensor([list(text.encode("utf-8"))], device=DEVICE)
    inp, target = ids[:, :-1], ids[:, 1:]
    opt.zero_grad()
    logits = model(inp)
    per_tok_loss = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1), reduction="none")
    per_tok_loss = per_tok_loss.view(1, -1)
    # target index i corresponds to predicting byte at position i+1 of the original text;
    # the color answer bytes start at prefix_len in the ORIGINAL text, so in target-space
    # (which is text[1:]) they start at prefix_len - 1.
    color_start = prefix_len - 1
    color_loss = per_tok_loss[:, color_start:].mean()
    rest_loss = per_tok_loss[:, :color_start].mean()
    total_loss = per_tok_loss.mean()
    total_loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"  step {step}: total={total_loss.item():.4f}  "
              f"color_bytes_loss={color_loss.item():.4f}  filler_bytes_loss={rest_loss.item():.4f}",
              flush=True)

print(f"\nFinal: total={total_loss.item():.4f}  color_bytes_loss={color_loss.item():.4f}  "
      f"filler_bytes_loss={rest_loss.item():.4f}")
print("\nIf color_bytes_loss stayed high while filler_bytes_loss dropped to near zero, "
      "that confirms the aggregate loss was hiding the real failure (model mastered "
      "predictable filler, never learned the actual copy-the-color task).")
