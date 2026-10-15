#!/usr/bin/env python3
"""THE ACTIVE-STATE FOLDING TEST, 2026-10-15 -- the gap flagged repeatedly and finally addressed:
does Lucy's persistent Mamba3 recurrent state actually retain information past the point the old
rolling-text-window mechanism (MAX_SESSION_TURNS=6, brickcluster_engine.py) structurally CANNOT.

This is a real "associative recall" task, the same shape used in actual SSM research (S4/H3/
Mamba papers) to test exactly this property -- not an invented methodology.

Fair comparison, stated precisely:
  - OLD mechanism: keeps only the last 6 turns. A fact stated more than 6 turns before the
    question is DELETED from its input -- not "harder to recall," literally absent. Success is
    mathematically impossible for it on this task, by construction.
  - NEW mechanism: ONE persistent Mamba3 instance processes the ENTIRE sequence (fact + filler +
    question) continuously via real recurrent state (.step()), never dropping anything from its
    input -- whether it SUCCEEDS depends on whether training actually taught it to use that
    capacity usefully, which is what this test measures.

Task: "My favorite color is X. [7+ unrelated filler sentences]. What is my favorite color? -> X"
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
N_FILLER_TURNS = 8   # > MAX_SESSION_TURNS=6 -- the old mechanism provably cannot see the fact
# by the time the question arrives, regardless of how good its generation is.

COLORS = ["blue", "red", "green", "yellow", "purple", "orange", "black", "white", "pink", "gray"]
FILLERS = [
    "The weather is nice today.", "I had cereal for breakfast.", "Traffic was busy this morning.",
    "My neighbor got a new car.", "The meeting ran long.", "I need to buy groceries later.",
    "The movie was pretty good.", "It rained a bit yesterday.", "The coffee shop was crowded.",
    "I finished reading a book.", "The dog needs a walk.", "My phone battery is low.",
]


def make_example(rng):
    color = rng.choice(COLORS)
    filler = " ".join(rng.sample(FILLERS, N_FILLER_TURNS))
    prefix = f"My favorite color is {color}. {filler} What is my favorite color? "
    text = prefix + color
    # 2026-10-15, real diagnosis via _diag_active_state_test.py: uniform next-byte loss over
    # the WHOLE sequence lets the model "win" on aggregate loss by mastering the easy,
    # highly-predictable filler text while NEVER learning the hard part (copying the color) --
    # filler_bytes_loss converged to ~0.11 while color_bytes_loss stayed at 0.75-1.4, never
    # converging, hidden behind a good-looking aggregate loss. Fix: return the prefix length so
    # training can weight loss toward the color-answer bytes specifically.
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
COLOR_LOSS_WEIGHT = 20.0  # upweight the bytes that actually matter so they can't be drowned
# out by trivially-easy filler-prediction loss, per the real diagnosis above.
print(f"Training on the associative-recall task ({N_FILLER_TURNS} filler sentences between "
      f"fact and question -- exceeds the old mechanism's 6-turn window), loss weighted "
      f"{COLOR_LOSS_WEIGHT}x on the color-answer bytes specifically...", flush=True)
losses = []
for step in range(N_TRAIN_STEPS):
    text, prefix_len = make_example(rng)
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
    losses.append(loss.item())
    if step % 100 == 0:
        color_loss = per_tok[:, color_start:].mean().item()
        print(f"  step {step}: weighted_loss={loss.item():.4f}  color_bytes_loss={color_loss:.4f}",
              flush=True)

print(f"\nFinal training loss: {losses[-1]:.4f} (started at {losses[0]:.4f})", flush=True)

# --- Held-out evaluation: NEW color/filler combinations never seen during training ---
print("\n=== HELD-OUT EVALUATION (fresh random examples) ===", flush=True)
eval_rng = random.Random(999)  # different seed -- genuinely unseen combinations
correct = 0
n_eval = 30
for _ in range(n_eval):
    color = eval_rng.choice(COLORS)
    filler = " ".join(eval_rng.sample(FILLERS, N_FILLER_TURNS))
    prompt = f"My favorite color is {color}. {filler} What is my favorite color? "
    gen = model.generate(prompt, n=len(max(COLORS, key=len)))
    answer = gen[len(prompt):]
    got_it = answer.strip().startswith(color)
    correct += int(got_it)
    print(f"  fact={color:8s} -> generated={answer[:12]!r}  {'OK' if got_it else 'WRONG'}", flush=True)

print(f"\n=== RESULT ===")
print(f"Held-out associative recall accuracy: {correct}/{n_eval} = {100*correct/n_eval:.1f}%")
print(f"(Filler length: {N_FILLER_TURNS} sentences, exceeding the old mechanism's "
      f"MAX_SESSION_TURNS=6 -- the old text-window approach is mathematically incapable of "
      f"answering this correctly by construction, since the fact would already be deleted "
      f"from its visible context by the time the question arrives.)")
