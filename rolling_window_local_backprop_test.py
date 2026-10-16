#!/usr/bin/env python3
"""THE ROLLING-WINDOW LOCAL BACKPROP TEST, 2026-10-15 -- Jonathan's own proven, already-designed
method (GALATEA_BOOK4_QUINARY_BRICKS_APPENDIX_A.txt, Scale 1: "the three-brick rolling window"),
adapted from a chained-bricks pipeline (A->B->C) to a single recurrent Mamba3 module: forward
information flow stays continuous across the whole sequence (via the real, verified .step()
incremental path -- bit-exact match to the full parallel scan, confirmed earlier today), but
gradients are TRUNCATED to a 3-step window -- error correction never propagates back further
than 3 steps before the persistent state is detached and the window slides forward with the
signal. This is real, standard truncated-BPTT (TBPTT), with the window deliberately set to 3 to
match the diary's exact specification, not a typical TBPTT window (usually 20-200).

Same favorite-color recall task as v3 (100% full-backprop baseline, checkpoint preserved) and
the Forward-Forward test (73.3% peak), for a direct three-way comparison.
"""
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"D:\mamba3_clean_ref")
import torch
import torch.nn as nn
import torch.nn.functional as F

# 2026-10-15, real correction: the official mamba_ssm.modules.mamba3.Mamba3's .step() requires
# H100-specific CuteDSL kernels (its own docstring: "Only tested on H100") and uses a different
# state signature (k_state/v_state, not Bx_prev_state) than what was verified earlier today.
# The bit-exact step-matches-forward verification this morning was on the CLEAN REFERENCE
# implementation specifically, not this official module -- using that verified one here instead,
# since it's pure PyTorch, hardware-portable, and already confirmed correct in step mode.
from mamba3 import Mamba3 as CleanMamba3

torch.manual_seed(0)
random.seed(0)
DEVICE = "cuda"
D_MODEL = 128
WINDOW = 3  # the diary's exact specification: "always three bricks deep"

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
        # SCOPE NOTE: this test isolates the local-vs-global backprop question specifically.
        # Full-precision parameters here (clean-reference Mamba3, no quinary wrapper), on
        # purpose, not a quiet drop of the quinary claim -- per-step quantization-with-grad
        # needs real additional plumbing (functional_call only binds a module's forward(), not
        # an arbitrary method like .step()), a separate, later task.
        self.mamba = CleanMamba3(d_model=d_model, d_state=32, expand=2, headdim=16, ngroups=1,
                                   is_mimo=False).to(DEVICE)
        self.head = nn.Linear(d_model, 256).to(DEVICE)

    def train_sequence(self, byte_ids: list[int], opt) -> float:
        """Processes the WHOLE sequence step by step via the real, verified incremental decode
        path. Forward state carries continuously across the full sequence (information flow is
        never cut). Every WINDOW steps, backward+optimizer-step on the accumulated local loss,
        then DETACH the persistent state -- gradients never reach further back than WINDOW steps,
        matching the diary's "C corrects B corrects A, then the window slides" exactly."""
        angle_state, ssm_state, bx_prev = self.mamba.allocate_inference_cache(1, device=DEVICE)
        total_loss = 0.0
        n_losses = 0
        window_loss = 0.0
        window_count = 0
        for t in range(len(byte_ids) - 1):
            inp_byte = torch.tensor([byte_ids[t]], device=DEVICE)
            target_byte = torch.tensor([byte_ids[t + 1]], device=DEVICE)
            x = self.emb(inp_byte)  # (1, d_model)
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
                # DETACH -- this is the actual truncation: the next window starts fresh,
                # carrying the real state VALUES forward (information keeps flowing) but with
                # no graph connection back to this window's computation (error correction
                # cannot reach past this point).
                angle_state = angle_state.detach()
                ssm_state = ssm_state.detach()
                bx_prev = bx_prev.detach()
                window_loss = 0.0
                window_count = 0

        if window_count > 0:
            opt.zero_grad()
            window_loss.backward()
            opt.step()

        return total_loss / max(n_losses, 1)

    @torch.no_grad()
    def score_completion(self, prompt: str, candidate: str) -> float:
        """Returns -loss (higher = better) for the candidate completion, host-off, via the same
        step-by-step path, non-quantized-with-grad (pure inference)."""
        full = prompt + candidate
        byte_ids = list(full.encode("utf-8"))
        prompt_len = len(prompt.encode("utf-8"))
        angle_state, ssm_state, bx_prev = self.mamba.allocate_inference_cache(1, device=DEVICE)
        total_nll = 0.0
        n = 0
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

N_TRAIN_EXAMPLES = 150  # step-by-step is much slower than parallel forward() -- real cost,
# already flagged as expected for local/regional methods.
print(f"Training with rolling-window local backprop (WINDOW={WINDOW}, step-by-step, truncated "
      f"gradient flow, real cost expected)...", flush=True)
for i in range(N_TRAIN_EXAMPLES):
    color = rng.choice(COLORS)
    text, _ = make_example(rng, color)
    byte_ids = list(text.encode("utf-8"))
    avg_loss = model.train_sequence(byte_ids, opt)
    if i % 15 == 0:
        print(f"  example {i}: avg_loss={avg_loss:.4f}", flush=True)

print(f"\n=== JONATHAN'S EXACT EXAMPLE ===", flush=True)
scores = {c: model.score_completion(JONATHANS_EXAMPLE, c) for c in COLORS}
pred = max(scores, key=scores.get)
print(f"  predicted: {pred}  (target: red)  {'OK' if pred == 'red' else 'WRONG'}", flush=True)

print("\n=== HELD-OUT EVALUATION (fresh random examples, all 10 colors) ===", flush=True)
eval_rng = random.Random(999)
correct = 0
n_eval = 30
for _ in range(n_eval):
    color = eval_rng.choice(COLORS)
    text, prefix_len = make_example(eval_rng, color)
    prompt = text[:prefix_len]
    scores = {c: model.score_completion(prompt, c) for c in COLORS}
    pred = max(scores, key=scores.get)
    got_it = pred == color
    correct += int(got_it)
    print(f"  fact={color:8s} -> predicted={pred:8s}  {'OK' if got_it else 'WRONG'}", flush=True)

print(f"\n=== RESULT ===")
print(f"Rolling-window (depth={WINDOW}) local backprop accuracy: {correct}/{n_eval} = {100*correct/n_eval:.1f}%")
print(f"(Full backprop baseline, v3: 100%. Forward-Forward peak: 73.3%.)")
