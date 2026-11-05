#!/usr/bin/env python3
"""Mamba3BrickCluster -- the real BrickCluster class (LSL-gated routing, growth, merge,
proven at F:\\my-agent-vampire-lucy, 71.1% retention result) with its ONLY change being which
brick type _spawn() creates. Routing/growth/merge are text-feature-based (TF-IDF trigram-hash)
and fully independent of brick internals, so this subclass changes NOTHING about the proven
gate logic -- only swaps FixedStepBrick (quinary gated-RNN) for Mamba3QuinaryBrickLM (real
Mamba-3 SISO, quinary-quantized)."""
import sys, os
sys.path.insert(0, r"F:\LucyWEST\engine")
sys.path.insert(0, os.path.dirname(__file__))

from brick_cluster_charlm import BrickCluster
from mamba3_brick_lm import Mamba3QuinaryBrickLM


class Mamba3BrickCluster(BrickCluster):
    def _spawn(self):
        idx = len(self.bricks)
        self.bricks.append(Mamba3QuinaryBrickLM(self.V, self.D_brick, seed=self.seed + 1000 * idx))
        from brick_cluster_charlm import _OnlineGaussian
        self.brick_pos.append(_OnlineGaussian())
        return idx

    @classmethod
    def load_state(cls, state):
        from brick_cluster_charlm import _OnlineGaussian
        n_bricks = int(state["n_bricks"])
        V = int(state["V"]); D_brick = int(state["D_brick"]); seed = int(state["seed"])
        # FIX (2026-11-05, real RAM-blowout root cause -- see PrivateLucyWEST's same fix /
        # BRICKCLUSTER_RAM_BLOWOUT_REPAIR_20261105.md for the full measured writeup): NpzFile's
        # .items()/.keys() are lazy with ZERO caching -- every call re-decompresses EVERY array in
        # the whole archive. The loop below calls state.items() once PER BRICK, so loading n_bricks
        # bricks did n_bricks full-archive decompressions (measured elsewhere: 402 bricks x 25,350
        # keys = 10.19M redundant reads for one checkpoint). Materializing once here turns that into
        # one real pass, total. No-op (shallow copy) if state is already a plain dict.
        state = dict(state.items())
        cluster = cls(V, D_brick, seed)
        cluster.neg.n = int(state["neg_n"])
        cluster.neg.sum = state["neg_sum"]
        cluster.neg.sumsq = state["neg_sumsq"]
        cluster.doc_freq = state["doc_freq"]
        cluster.n_docs = int(state["n_docs"])
        for i in range(n_bricks):
            brick = Mamba3QuinaryBrickLM(V, D_brick, seed=seed + 1000 * i)
            prefix = f"brick{i}__"
            bstate = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            brick.load_state(bstate)
            cluster.bricks.append(brick)
            pos = _OnlineGaussian()
            pos.n = int(state[f"pos{i}__n"])
            pos.sum = state[f"pos{i}__sum"]
            pos.sumsq = state[f"pos{i}__sumsq"]
            cluster.brick_pos.append(pos)
        return cluster
