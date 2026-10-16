#!/usr/bin/env python3
"""PC-ALM (Augmented Lagrangian Predictive Coding, Sakana AI, arXiv:2605.31022), faithfully
implemented for a 4-brick chain (reusing the chain structure from the earlier true-pipeline
test), 2026-10-15. Real algorithm, verified from the paper's own writeup:

  r_i = h_i - brick_i(h_{i-1})                              (local prediction error)
  h_i <- h_i - eta_h * (r_i - J_{i+1}^T(r_{i+1}) + lambda_i) (inference, T iters, autograd VJP
                                                               generalizes W^T(sigma' . r) to
                                                               Mamba3's real nonlinear recurrence)
  lambda_i <- lambda_i + alpha * r_i                         (dual/Lagrange multiplier update)
  W_i <- W_i + eta_theta * autograd_grad(brick_i, r_i)       (ONE weight step after inference
                                                               settles, Hebbian-shaped)

NO backprop through the whole chain anywhere -- h_{i-1} is detached before each brick_i call;
all cross-layer credit flows through the autograd vector-Jacobian product of ONE brick at a
time (r_{i+1} as grad_output), matching the paper's "layer-local, neighbor-only" claim exactly.

2026-10-15, POST-MORTEM ON THE FIRST RUN, two real bugs found and fixed here:
1. SPEED: the first run used D:\mamba3_clean_ref's pure-Python-loop reference Mamba3 (its own
   docstring: "correct but slow... for research/educational purposes"). Measured directly:
   ~0.74s/forward-call at this task's real L=482 sequence length. PC-ALM's design calls that
   ~25-30 times per training step, so one step cost ~80s -- 400 steps would have taken 9+ hours.
   Switched to the REAL official `mamba_ssm.Mamba3` (Triton kernels, verified running on this
   RTX 4090 earlier this session for the production 82.7% result). Measured after JIT warmup:
   0.0046s/call forward-only, 0.0125s/call forward+backward -- a ~100-150x speedup. No .step()
   (incremental decode) is used anywhere in PC-ALM, only full-sequence forward(), so the earlier
   H100-only CuteDSL limitation (which only affects .step()) does not apply here.
2. CORRECTNESS: the first run went to NaN by step 50 and silently stayed NaN through step 250+
   (confirmed from the buffered output after killing it) -- the local Hebbian weight update
   (`p += ETA_THETA * autograd_grad(...)`) is unbounded; nothing capped the residual magnitude
   feeding it, so a runaway positive-feedback blowup (bigger r_i -> bigger weight step -> bigger
   r_i next time) was free to happen. Fixed with a residual clamp (bounds the signal feeding the
   inference update, dual update, AND weight update all at once, since all three read the same
   r_i) and a NaN guard that stops immediately instead of burning hours on a dead run.
3. SECOND SPEED BUG, found while verifying the first fix: Triton's real kernel is
   `@triton.autotune`'d -- it compiles AND benchmarks several config candidates the first time
   it sees a given (shape, dtype) combination, then caches the winner. Training examples here
   have VARIABLE length (different sentence counts / color-word lengths), so every new length
   was triggering a fresh 30-90s autotune-compile cycle, observed directly via py-spy showing the
   process inside `triton.runtime.autotuner.benchmark` on the very first training step. Fixed by
   padding every example to one FIXED length (MAX_LEN) so Triton autotunes exactly once for the
   whole run, with the loss masked to zero on the padding so it contributes nothing to training.
4. SECOND CORRECTNESS BUG, found immediately after re-running with the real (differently
   initialized) Mamba3 module -- went NaN on literally step 1 this time, worse than the first
   run. Clamping r_i alone was not enough: `pred` itself (unbounded brick output) still feeds the
   weight-update gradient directly (`loss_i = (pred * r_i).sum()`), and the VJP term / dual-update
   term feeding the INFERENCE update were never clamped at all -- `h_i` could already diverge
   inside the T_INFERENCE loop, before any weight update even happens. Added: (a) clamp vjp_term
   and grad_last the same way r_i is clamped, (b) clamp h_i itself after every inference update
   (standard practice in real predictive-coding implementations, to keep inference dynamics in a
   sane range), (c) global-norm gradient clipping on the per-brick weight-update grads.
5. THIRD BUG, found via direct diagnostic after a tuning pass produced byte-for-byte IDENTICAL
   output to the first clean run despite a 10x larger ETA_THETA -- the brick parameters were
   barely moving at all (measured: total param norm ~28.19 -> ~28.19 over 25 steps). Root cause,
   confirmed by printing the raw per-step gradient norm before clipping: the local Hebbian
   objective's gradient magnitude is wildly non-stationary, varying over 4-6 orders of magnitude
   step to step (observed: 6738 at step 0, 0.0008 at step 5, 5003 at step 15, back to 0.0006 at
   step 20). `GRAD_CLIP_NORM` only intervenes on the LARGE end -- on the many steps where the raw
   gradient happened to be tiny, the clip never engaged and `ETA_THETA * tiny_grad` was
   functionally zero, so only the rare huge-gradient steps produced any real movement. Fixed by
   normalizing the gradient DIRECTION and taking a fixed-magnitude step every time (standard
   normalized-gradient-ascent, not norm-clipping), so every step contributes equally regardless
   of the raw gradient's scale.
6. FOURTH BUG: even after bug 5's fix, bricks were genuinely moving (confirmed: the loss
   trajectory changed shape, no longer identical to the un-moving baseline) but STILL produced
   byte-for-byte identical, 0/30 output. Root cause: the SAME loss-dilution pattern already found
   and fixed twice earlier tonight (Forward-Forward's goodness-over-whole-sequence, the
   active-state filler-dilution bug) -- `loss_i = (pred * r_i).sum()` summed over all 512
   positions UNIFORMLY, so the ~480 bytes of easy, predictable filler template text completely
   drowned the ~5 bytes of color-word that actually matter. The supervised head loss already had
   `loss_weights` (20x on the color bytes) from the start; the brick's own local Hebbian loss
   never did. Fixed by applying the same `loss_weights` mask to the local loss.
7. FIFTH BUG: still 0/30 with a consistent oscillating "o"/space output pattern. Found that
   `generate()` did a plain one-shot feedforward pass through the bricks with NO inference
   settling at all -- a real mismatch, since during TRAINING the head only ever saw hs[-1]
   AFTER the full T_INFERENCE settling loop pulled it toward the target. Real predictive-coding
   protocols run inference at test time too (just without target supervision, since the byte
   being predicted is unknown). Fixed by making generate() run the same settle_hs() loop, with
   loss_weights=0 at the one position being predicted (no label to supervise with there) so it
   settles on local/VJP consistency alone, and loss_weights=1 on the known teacher-forced prefix.
8. SIXTH BUG, the deepest one, found while re-verifying bug 7's fix changed little: `emb`
   (`h0 = emb(byte_ids)`) was computed inside `with torch.no_grad():` at the very top of
   settle_hs -- meaning the embedding layer NEVER received a gradient, in ANY run tonight
   (v4 through v8), including before the Triton swap. It was frozen at random initialization
   the entire time; only `head`'s linear readout and the bricks' own Mamba3 parameters could
   ever adapt. The model was trying to solve the task through fixed, arbitrary, never-learned
   byte representations the whole time -- plausibly sufficient on its own to explain total
   failure. Fixed by giving `emb` its own local update, the same one-hop-VJP pattern used
   between bricks (treating emb as "layer 0" feeding brick 0), rather than silently relying on
   an `io_opt` path that never actually reached it.
"""
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba3 as RealMamba3

torch.manual_seed(0)
random.seed(0)
DEVICE = "cuda"
D_MODEL = 64
N_BRICKS = 4
T_INFERENCE = 8        # paper: T ~ proportional to depth
ETA_H = 0.3
ALPHA = 0.1
R_CLAMP = 5.0          # real bug fix: nothing bounded residual magnitude before; this does
H_CLAMP = 20.0         # bounds h_i itself during inference -- r_i alone wasn't enough, hs
                       # could still diverge inside the T_INFERENCE loop before any weight step
STEP_NORM = 0.05       # fixed per-step update magnitude (normalized-gradient-ascent), replacing
                       # norm-clipping -- the raw gradient's scale varies over orders of
                       # magnitude step to step, so clipping-only left most steps as no-ops
MAX_LEN = 512          # fixed length for ALL sequences (measured real max example: 493 bytes)
                       # -- one shape for Triton's autotuner to compile once, not every step
PAD_BYTE = 0

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


emb = nn.Embedding(256, D_MODEL).to(DEVICE)
head = nn.Linear(D_MODEL, 256).to(DEVICE)
bricks = [RealMamba3(d_model=D_MODEL, d_state=24, headdim=16, is_mimo=False,
                      dtype=torch.float32).to(DEVICE) for _ in range(N_BRICKS)]
io_opt = torch.optim.Adam(list(head.parameters()), lr=3e-3)  # emb now gets its own local
                                                               # update (see pc_alm_step) --
                                                               # it never had a gradient path
                                                               # through this optimizer at all

print(f"Warming up Triton kernels at the real fixed shape (L={MAX_LEN}) so autotuning happens "
      f"exactly once, not mid-training...", flush=True)
with torch.no_grad():
    _warm = torch.randn(1, MAX_LEN, D_MODEL, device=DEVICE)
    for b in bricks:
        b(_warm)


def pad_to_fixed(text: str, color_start: int):
    """Pads text to MAX_LEN+1 bytes (PAD_BYTE), returns (inp, target, weights) all length
    MAX_LEN -- weights is the existing color-emphasis mask, now also zeroing out padding."""
    raw = list(text.encode("utf-8"))
    real_len = len(raw)
    assert real_len <= MAX_LEN + 1, f"example length {real_len} exceeds MAX_LEN+1={MAX_LEN + 1}"
    padded = raw + [PAD_BYTE] * (MAX_LEN + 1 - real_len)
    ids = torch.tensor([padded], device=DEVICE)
    inp, target = ids[:, :-1], ids[:, 1:]
    weights = torch.ones(1, MAX_LEN, device=DEVICE)
    weights[:, color_start - 1:real_len - 1] = 20.0   # real color-answer bytes, emphasized
    weights[:, real_len - 1:] = 0.0                    # padding -- contributes nothing
    return inp, target, weights


def settle_hs(byte_ids: torch.Tensor, target_ids: torch.Tensor, loss_weights: torch.Tensor):
    """Runs the T_INFERENCE-iteration local settling loop and returns the settled hs list.
    Shared by training AND generation -- real predictive-coding protocols run inference at
    BOTH train and test time; only the WEIGHT update is train-only. The caller controls what's
    "known" purely through loss_weights: zero weight at a position means no supervision there,
    settling for that position then runs on pure local/VJP consistency alone (used by generate()
    for the one byte that's actually unknown)."""
    with torch.no_grad():
        h0 = emb(byte_ids)
        # Feedforward init for all hidden states.
        hs = [h0]
        cur = h0
        for brick in bricks:
            cur = brick(cur)
            hs.append(cur)
    # hs[0] = input (clamped, never updated). hs[1..N_BRICKS] are free variables during inference.
    hs = [hs[0]] + [h.clone().requires_grad_(False) for h in hs[1:]]
    lambdas = [torch.zeros_like(h) for h in hs[1:]]

    for t in range(T_INFERENCE):
        preds = [None] * (N_BRICKS + 1)
        rs = [None] * (N_BRICKS + 1)
        # forward predictions, each from its own (detached) input -- no cross-layer backprop.
        for i in range(N_BRICKS):
            inp = hs[i].detach().requires_grad_(True)
            pred = bricks[i](inp)
            preds[i + 1] = pred
            rs[i + 1] = (hs[i + 1] - pred).clamp(-R_CLAMP, R_CLAMP)
        # output-side signal: gradient of the supervised loss w.r.t. the LAST hidden state,
        # standing in for "r_{L+1}" in the paper's formulation (output layer has no free h_L,
        # it connects straight to the loss).
        last_inp = hs[-1].detach().requires_grad_(True)
        logits = head(last_inp)
        per_tok = F.cross_entropy(logits.reshape(-1, 256), target_ids.reshape(-1),
                                   reduction="none").view(1, -1)
        sup_loss = (per_tok * loss_weights).sum() / loss_weights.sum()
        (grad_last,) = torch.autograd.grad(sup_loss, last_inp)
        grad_last = grad_last.clamp(-R_CLAMP, R_CLAMP)

        new_hs = [hs[0]]
        for i in range(1, N_BRICKS + 1):
            inp_i_minus_1 = hs[i - 1].detach().requires_grad_(True)
            pred_i = bricks[i - 1](inp_i_minus_1)
            r_i = (hs[i] - pred_i).detach().clamp(-R_CLAMP, R_CLAMP)
            if i < N_BRICKS:
                # J_{i+1}^T(r_{i+1}) via autograd VJP on brick i+1's own forward.
                inp_next = hs[i].detach().requires_grad_(True)
                pred_next = bricks[i](inp_next)
                (vjp_term,) = torch.autograd.grad(pred_next, inp_next, grad_outputs=rs[i + 1])
                vjp_term = vjp_term.clamp(-R_CLAMP, R_CLAMP)
            else:
                vjp_term = grad_last  # last interior layer feeds directly from the supervised loss
            h_i_new = hs[i] - ETA_H * (r_i - vjp_term + lambdas[i - 1])
            h_i_new = h_i_new.clamp(-H_CLAMP, H_CLAMP)
            new_hs.append(h_i_new.detach())
            lambdas[i - 1] = lambdas[i - 1] + ALPHA * r_i
        hs = new_hs
    return hs


def pc_alm_step(byte_ids: torch.Tensor, target_ids: torch.Tensor, loss_weights: torch.Tensor):
    hs = settle_hs(byte_ids, target_ids, loss_weights)

    # ONE weight update per brick, using the SETTLED r_i and h_{i-1}, local only.
    for i in range(N_BRICKS):
        inp = hs[i].detach().requires_grad_(True)
        pred = bricks[i](inp)
        r_i = (hs[i + 1] - pred).detach().clamp(-R_CLAMP, R_CLAMP)
        # same loss-dilution bug already found+fixed twice tonight (FF goodness, active-state
        # filler): weight by task relevance or the ~500-byte filler region drowns the one byte
        # that matters. loss_weights is (1, MAX_LEN); broadcast over the D_MODEL dim.
        loss_i = (pred * r_i * loss_weights.unsqueeze(-1)).sum()
        grads = torch.autograd.grad(loss_i, list(bricks[i].parameters()), retain_graph=False,
                                     allow_unused=True)
        with torch.no_grad():
            total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None))
            scale = STEP_NORM / (total_norm + 1e-6)  # normalize direction, fixed step magnitude
            for p, g in zip(bricks[i].parameters(), grads):
                if g is not None:
                    p += g * scale

    # 2026-10-15, REAL BUG FOUND AND FIXED: emb never received ANY gradient in any run tonight
    # (v4-v8) -- h0=emb(byte_ids) was computed inside `with torch.no_grad()` at the very top of
    # settle_hs, so the embedding was frozen at random init the entire time; only head and the
    # bricks could ever adapt. Gives emb its own local update, same one-hop-VJP pattern as a
    # brick (treating emb as "layer 0" feeding brick 0), using the settled hs[1] as the target
    # brick 0's prediction should match.
    h0_g = emb(byte_ids)
    pred0 = bricks[0](h0_g)
    r_1_for_emb = (hs[1] - pred0).detach().clamp(-R_CLAMP, R_CLAMP)
    loss_emb = (pred0 * r_1_for_emb * loss_weights.unsqueeze(-1)).sum()
    grads_emb = torch.autograd.grad(loss_emb, list(emb.parameters()), retain_graph=False,
                                     allow_unused=True)
    with torch.no_grad():
        total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads_emb if g is not None))
        scale = STEP_NORM / (total_norm + 1e-6)
        for p, g in zip(emb.parameters(), grads_emb):
            if g is not None:
                p += g * scale

    # head still trained by ordinary backprop -- it's not part of the chain's
    # local-credit claim, same convention as treating them as the fixed "sensor/readout" stage.
    io_opt.zero_grad()
    logits = head(hs[-1])
    per_tok = F.cross_entropy(logits.reshape(-1, 256), target_ids.reshape(-1),
                               reduction="none").view(1, -1)
    final_loss = (per_tok * loss_weights).sum() / loss_weights.sum()
    final_loss.backward()
    io_opt.step()
    return final_loss.item()


rng = random.Random(0)
N_STEPS = 800          # 2026-10-15 tuning pass: doubled now that each step is ~0.2s, not ~80s
print(f"Training PC-ALM, {N_BRICKS} bricks, T={T_INFERENCE} inference iterations/step...", flush=True)
t_start = __import__("time").time()
for step in range(N_STEPS):
    color = rng.choice(COLORS)
    text, prefix_len = make_example(rng, color)
    inp, target, loss_weights = pad_to_fixed(text, prefix_len)
    loss = pc_alm_step(inp, target, loss_weights)
    if loss != loss:  # NaN check -- the first run went NaN at step 50 and burned 4+ hours unnoticed
        print(f"  step {step}: loss=NaN -- STOPPING, do not burn more compute on a dead run",
              flush=True)
        sys.exit(1)
    if step % 25 == 0:
        elapsed = __import__("time").time() - t_start
        print(f"  step {step}: loss={loss:.4f}  ({elapsed:.1f}s elapsed)", flush=True)


def generate(prompt: str, n: int) -> str:
    """2026-10-15, real fix: this used to be a plain feedforward pass through the bricks with
    NO inference settling at all -- a mismatch with training, where the head only ever saw
    hs[-1] AFTER the full T_INFERENCE settling loop pulled it toward the target. Real
    predictive-coding protocols run inference at test time too, just without the target
    supervision (since the byte being predicted is unknown). Now uses the SAME settle_hs() loop:
    known prefix bytes get normal supervision (teacher-forced, loss_weights=1), the ONE position
    being predicted gets loss_weights=0 (no target to supervise with -- it settles on local/VJP
    consistency alone), matching exactly what training actually optimized the weights for."""
    ids = list(prompt.encode("utf-8"))
    for _ in range(n):
        real_len = len(ids)
        padded = ids + [0] + [PAD_BYTE] * (MAX_LEN - real_len)  # total MAX_LEN+1, same
                                                                  # convention as pad_to_fixed
        full_ids = torch.tensor([padded], device=DEVICE)
        inp, target = full_ids[:, :-1], full_ids[:, 1:]
        loss_weights = torch.zeros(1, MAX_LEN, device=DEVICE)
        loss_weights[:, :real_len - 1] = 1.0   # known prefix: normal teacher-forced supervision
        # position real_len-1 (predicts the next, unknown byte) stays weight=0 on purpose
        # NOTE: settle_hs needs autograd enabled internally for its VJP calls -- no_grad() here
        # would break it; we just never call .backward() on anything built during generation.
        hs = settle_hs(inp, target, loss_weights)
        with torch.no_grad():
            logits = head(hs[-1])
        nxt = int(torch.argmax(logits[0, real_len - 1]).item())
        ids.append(nxt)
    return bytes(ids[len(prompt.encode("utf-8")):]).decode("utf-8", errors="replace")


print("\n=== JONATHAN'S EXACT EXAMPLE ===", flush=True)
answer = generate(JONATHANS_EXAMPLE, n=6)
print(f"  generated: {answer!r}  (target: red)  {'OK' if answer.strip().startswith('red') else 'WRONG'}",
      flush=True)

print("\n=== HELD-OUT EVALUATION ===", flush=True)
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
print(f"PC-ALM ({N_BRICKS} bricks) accuracy: {correct}/{n_eval} = {100*correct/n_eval:.1f}%")
print(f"(Full backprop: 100%. Forward-Forward: 73.3%. TBPTT(3): 3.3%. True-chain: 0%. "
      f"Sister-split: 93.3% but no effect vs ablation.)")
