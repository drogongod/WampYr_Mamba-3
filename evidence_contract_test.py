#!/usr/bin/env python3
"""THE NON-NEGOTIABLE EXPERIMENT, F:\\Wampyr.txt, applied to the real Mamba3QuinaryBrick.

Same evidence-contract shape as F:\\my-agent-vampire-lucy\\evidence_contract_test.py, now against
a REAL Mamba-3 recurrence (official Triton-kernel mamba_ssm.modules.mamba3.Mamba3) with every
trainable parameter quinary-quantized (STE), instead of the quinary gated-RNN brick.

1. Ask X (greedy-generate the continuation). Confirm the brick does NOT produce it, host-off.
2. Teach X (real backprop through the quantized forward -- next-byte prediction loss).
3. Ask X again. Verify the brick NOW produces the correct continuation, host-off.
4. Teach unrelated Y and Z.
5. Ask X again. Verify retention survived teaching unrelated facts (not clobbered).

Generation, not loss, is the evidence -- per Wampyr.txt: "Loss going down alone is NOT proof."
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba3_quinary_brick import Mamba3QuinaryBrick

torch.manual_seed(0)
DEVICE = "cuda"
D_MODEL = 128


class ByteLM(nn.Module):
    """Minimal byte-level char-LM wrapper around the quinary-quantized Mamba3 brick.
    Embedding/output-head stay full precision (small, not the substrate under test);
    the Mamba3 recurrence itself is the thing being evidenced as quinary-native."""

    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(256, d_model).to(DEVICE)
        self.brick = Mamba3QuinaryBrick(d_model=d_model, d_state=32, headdim=16, device=DEVICE)
        self.head = nn.Linear(d_model, 256).to(DEVICE)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(byte_ids)
        h = self.brick.quantized_forward_with_grad(x)
        return self.head(h)

    @torch.no_grad()
    def generate(self, prompt: str, n: int = 20) -> str:
        ids = list(prompt.encode("utf-8"))
        for _ in range(n):
            x = torch.tensor([ids], device=DEVICE)
            logits = self.forward(x)
            next_id = int(torch.argmax(logits[0, -1]).item())
            ids.append(next_id)
        return bytes(ids).decode("utf-8", errors="replace")


def train_on_text(model, opt, text, steps):
    ids = torch.tensor([list(text.encode("utf-8"))], device=DEVICE)
    inp, target = ids[:, :-1], ids[:, 1:]
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def check(model, prompt, expected_continuation, label):
    gen = model.generate(prompt, n=len(expected_continuation.encode("utf-8")))
    got_continuation = gen[len(prompt):]
    matched = got_continuation.strip().startswith(expected_continuation.strip()[:6])
    print(f"  [{label}] prompt={prompt!r} -> generated={gen!r} "
          f"expected_cont={expected_continuation!r} MATCH={matched}")
    return matched


FACT_X_PROMPT = "The capital of England is"
FACT_X_FULL = "The capital of England is London."
FACT_Y_FULL = "The capital of France is Paris."
FACT_Z_FULL = "The capital of Germany is Berlin."

model = ByteLM()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

print("n_quinary_params in brick:", model.brick.n_quinary_params())

print("\n=== STEP 1: ask X before teaching ===")
before = check(model, FACT_X_PROMPT, " London.", "BEFORE")
assert not before, "should NOT know X before teaching (if it does, test is invalid)"

print("\n=== STEP 2: teach X ===")
losses_x = train_on_text(model, opt, FACT_X_FULL, steps=150)
print(f"  loss: {losses_x[0]:.3f} -> {losses_x[-1]:.3f}")

print("\n=== STEP 3: ask X again, host-off ===")
after_x = check(model, FACT_X_PROMPT, " London.", "AFTER TEACHING X")

print("\n=== STEP 4: teach unrelated Y and Z ===")
losses_y = train_on_text(model, opt, FACT_Y_FULL, steps=150)
losses_z = train_on_text(model, opt, FACT_Z_FULL, steps=150)
print(f"  Y loss: {losses_y[0]:.3f} -> {losses_y[-1]:.3f}")
print(f"  Z loss: {losses_z[0]:.3f} -> {losses_z[-1]:.3f}")

print("\n=== STEP 5: ask X again -- retention under interference ===")
retained_x = check(model, FACT_X_PROMPT, " London.", "AFTER TEACHING Y,Z")

print("\n=== RESULT ===")
print(f"  Knew X before teaching:            {before}  (must be False)")
print(f"  Knew X right after teaching:       {after_x}")
print(f"  Retained X after unrelated Y,Z:    {retained_x}")
print(f"  REAL MAMBA-3 QUINARY EVIDENCE CONTRACT: "
      f"{'PASS' if (not before and after_x and retained_x) else 'FAIL'}")
