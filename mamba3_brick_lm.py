#!/usr/bin/env python3
"""Mamba3QuinaryBrickLM -- matches the exact external interface BrickCluster expects from a
brick (learn_batch, forward_logits, generate, export_state, load_state, .V, .D), so it is a
real drop-in replacement for FixedStepBrick, internally backed by the real Mamba3 SISO block
(official Triton kernels) with every trainable parameter quinary-quantized via STE."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba3_quinary_brick import Mamba3QuinaryBrick, quinary_quant

DEVICE = "cuda"


class Mamba3QuinaryBrickLM:
    def __init__(self, V, D, seed):
        self.V = V
        self.D = D
        self.seed = seed
        torch.manual_seed(seed)
        self.emb = nn.Embedding(V, D).to(DEVICE)
        self.brick = Mamba3QuinaryBrick(d_model=D, d_state=32, headdim=16, device=DEVICE)
        self.head = nn.Linear(D, V).to(DEVICE)
        params = list(self.emb.parameters()) + list(self.brick.mamba.parameters()) + list(self.head.parameters())
        self.opt = torch.optim.Adam(params, lr=3e-3)

    def _forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        h = self.brick.quantized_forward_with_grad(x)
        return self.head(h)

    def learn_batch(self, idxs_batch, lr_scale=1.0):
        ids = torch.tensor(idxs_batch, dtype=torch.long, device=DEVICE)
        inp, target = ids[:, :-1], ids[:, 1:]
        for g in self.opt.param_groups:
            g["lr"] = 3e-3 * lr_scale
        self.opt.zero_grad()
        logits = self._forward(inp)
        per_tok = F.cross_entropy(logits.reshape(-1, self.V), target.reshape(-1), reduction="none")
        per_seq = per_tok.view(ids.shape[0], -1).mean(dim=1)
        per_seq.mean().backward()
        self.opt.step()
        return per_seq.detach().cpu().numpy()

    def forward_logits(self, idxs):
        with torch.no_grad():
            ids = torch.tensor([idxs], dtype=torch.long, device=DEVICE)
            logits = self._forward(ids)
        return logits[0].cpu().numpy()

    def generate(self, seed_idx, itos, n=60, temp=0.7, rep_pen=1.0, seed=0):
        torch.manual_seed(seed)
        # seed_idx may be a single starting index OR a full seed-prompt list of indices
        # (BrickCluster.generate passes a whole encoded prompt as "seed_idx" -- see its
        # own call site, LI.enc(seed_text)[-121:]).
        ids = list(seed_idx) if isinstance(seed_idx, (list, tuple)) else [seed_idx]
        n_seed = len(ids)
        with torch.no_grad():
            for _ in range(n):
                t = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                logits = self._forward(t)[0, -1]
                if temp > 0:
                    probs = F.softmax(logits / temp, dim=-1)
                    nxt = int(torch.multinomial(probs, 1).item())
                else:
                    nxt = int(torch.argmax(logits).item())
                ids.append(nxt)
        return "".join(itos[i] for i in ids[n_seed:])

    @property
    def Win(self) -> np.ndarray:
        """Compatibility shim for brickcluster_engine.py's WEIGHT_ARRAY_NAMES diffing (built for
        FixedStepBrick's named numpy level-arrays) -- ALL of this brick's quinary-quantized
        levels, concatenated into one int8 array, recomputed fresh on every access so a
        before/after snapshot genuinely reflects current state, not a stale copy."""
        with torch.no_grad():
            levels = [torch.round(quinary_quant(p) / p.abs().mean().clamp(min=1e-8)).to(torch.int8)
                      for p in self.brick.mamba.parameters()]
            flat = torch.cat([l.flatten() for l in levels])
        return flat.cpu().numpy()

    def export_state(self):
        state = {
            "emb": self.emb.weight.detach().cpu().numpy(),
            "head_w": self.head.weight.detach().cpu().numpy(),
            "head_b": self.head.bias.detach().cpu().numpy(),
        }
        for k, v in self.brick.mamba.state_dict().items():
            state[f"mamba__{k}"] = v.detach().cpu().numpy()
        return state

    def load_state(self, state):
        self.emb.weight.data = torch.tensor(np.asarray(state["emb"]), device=DEVICE)
        self.head.weight.data = torch.tensor(np.asarray(state["head_w"]), device=DEVICE)
        self.head.bias.data = torch.tensor(np.asarray(state["head_b"]), device=DEVICE)
        sd = {}
        prefix = "mamba__"
        keys = state.files if hasattr(state, "files") else state.keys()
        for k in keys:
            if k.startswith(prefix):
                sd[k[len(prefix):]] = torch.tensor(np.asarray(state[k]), device=DEVICE)
        self.brick.mamba.load_state_dict(sd)
