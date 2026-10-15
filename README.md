# WampYr_Mamba-3

A real, working Mamba-3 integration of the [WampYr](https://github.com/drogongod/WampYr) vampire
substrate: own-weights-first, donor-fallback continuous learning, now with genuine Mamba-3
recurrence (official Triton-kernel implementation, Lahoti/Li/Chen/Wang/Bick/Kolter/Dao/Gu,
arXiv:2603.15569) instead of the gated-RNN recurrence used in the original WampYr.

## What's actually real here, stated plainly

- **Real Mamba-3**: `mamba_ssm.modules.mamba3.Mamba3`, the official SISO implementation, real
  Triton kernels (`mamba3_siso_combined`), exponential-trapezoidal discretization, RoPE-based
  complex state. Verified running correctly on this project's own hardware (an RTX 4090 laptop,
  Windows) via the library's own built-in tests before anything was built on top of it.
- **Native quinary** (`mamba3_quinary_brick.py`): every trainable parameter of the Mamba3 block
  -- not just the two big projection matrices -- is quantized to 5 discrete levels {-2,-1,0,1,2}
  via STE (straight-through estimator) quantization-aware training. A float32 master copy exists
  only for the optimizer; the real Mamba3 forward/backward (unmodified, real Triton kernels) runs
  on the quantized values via `torch.func.functional_call`.
- **The proven routing/growth/merge gate, completely unmodified** (`mamba3_brick_cluster.py`):
  this only swaps which brick type gets spawned (`Mamba3QuinaryBrickLM` instead of
  `FixedStepBrick`). Routing is text-feature-based (TF-IDF trigram-hash), entirely independent of
  brick internals -- confirmed directly by running the identical 3-fact test against both brick
  types and getting bit-for-bit identical margin values.

## Real, measured result

Full 1000-fact teaching run (`amelia_teaches_lucy_mamba3.py`), same methodology as the original
WampYr's proven run (Amelia-voiced donor teacher, auto-accept, periodic consolidation,
MERGE_SLACK=0.08): **827/1000 = 82.7% retained host-off**, 189 final bricks, 1860 total seen.

This beats the original gated-RNN substrate's own best result (711/1000 = 71.1%, same scale, same
methodology) by +11.6 percentage points -- the first direct, apples-to-apples comparison between
the two recurrence types at real scale.

## Honest limits

- MIMO mode (the decode-throughput optimization in the real Mamba-3 paper) is not used here --
  it depends on `tilelang`, which has a packaging bug in its current PyPI release and is a
  Linux-first build toolchain. SISO (used throughout) has the same defining properties
  (trapezoidal discretization, RoPE complex state, the folding memory mechanism) -- MIMO is a
  throughput optimization, not a different architecture.
- At very small sample counts (tested directly: 3 facts total), the margin-based routing gate is
  genuinely unstable -- confirmed to be a property of the gate itself (identical behavior,
  identical margin values, regardless of brick type), not something introduced by this
  integration.
- `evidence_contract_test_original_gatedrnn.py` (the side-by-side comparison script) depends on
  the sibling WampYr repo being present at a fixed path -- not a standalone dependency of this
  repo's own pipeline.

## Files

| File | Role |
|---|---|
| `mamba3_quinary_brick.py` | The real Mamba3 SISO block wrapped with full-parameter STE quinary quantization. |
| `mamba3_brick_lm.py` | Byte-level char-LM wrapper matching BrickCluster's brick interface (`learn_batch`, `forward_logits`, `generate`, `export_state`/`load_state`). |
| `mamba3_brick_cluster.py` | Subclasses the proven `BrickCluster`, swapping only which brick type gets spawned. |
| `mamba3_lutgtlm_ingest.py` | Mamba3 equivalent of the proven ingest module (byte-level encoding, checkpoint save/load). |
| `amelia_teaches_lucy_mamba3.py` | The teaching-loop script that produced the 82.7% result above. |
| `evidence_contract_test.py` | Single-brick evidence test (catastrophic forgetting demonstrated directly, motivating the clustered test). |
| `evidence_contract_test_clustered.py` | Multi-fact test through real routing. |
| `evidence_contract_test_original_gatedrnn.py` | Side-by-side comparison against the original gated-RNN bricks. |
| `evidence_contract_test_programming_agent.py` | Real programming task through the donor-fallback mechanism, donor code verified by actual execution, not eyeballed. |
| `train_coding_dataset.py` | Direct teaching from a real, pre-answered coding dataset (no donor round-trip needed). |
| `mamba3_quinary_pack.py` | Pack-for-inference export: compact on-disk quinary storage (nibble-packed), dequantize-on-load. Verified lossless round trip. |
| `active_state_folding_test*.py` | The session-memory/active-state folding experiments (see below). |

## More real results

**Programming-agent mechanism**: donor-fallback verified working for actual code, not just trivia
-- a real `reverse_string` task, donor-generated code verified correct by **actually executing
it** against 4 test cases, learned immediately. (Retention-under-interference at this task alone
is inconclusive at n=2 facts -- same small-sample gate limit documented above, not yet tested at
scale for code specifically within the donor-fallback path.)

**Direct dataset training, real scale**: teaching directly from a real, pre-verified coding
dataset's own (instruction, output) pairs (no donor needed) --
- 500 examples: **456/500 = 91.2%** retained host-off, 428 bricks.
- 2000 examples: **1832/2000 = 91.6%**, holding steady at 4x scale -- and a real phase-transition
  consolidation event at fact 1000 (429 bricks merged in one pass, 779->350), after which the
  brick population actually *shrank* despite 4x more data taught, ending at a smaller footprint
  (271 bricks) than the 500-example run (428).

**Active-state / session-memory folding (the gap distinct from long-term factual retention)**:
tested via a real associative-recall benchmark (same shape as the actual S4/H3/Mamba academic
benchmark for this property) -- a fact stated once, then enough filler to exceed the old
rolling-text-window's `MAX_SESSION_TURNS=6`, then asked for. The old mechanism cannot answer this
by construction (the fact is deleted from its input); whether the real Mamba3 recurrence does is
an empirical question about training, not architecture.
- Sparse single mention + unrelated filler: 6.7% held-out, mode-collapsed to one fixed wrong
  answer. Diagnosed via a metric split: filler-bytes loss converged, color-bytes loss never did
  -- the aggregate loss was hiding total failure on the one thing that mattered.
- Reinforced density (target repeated 5-8x across varied sentences): 93.3% held-out, but a
  specific repeatable confusion (orange/green) and failure to generalize to genuinely novel
  phrasing outside the trained template family.
- Direct co-occurrence (the two words that actually matter -- "favorite" and the color --
  repeated together, not just the color alone): **100% held-out**, the orange/green confusion
  resolved, and generalization to genuinely novel out-of-template phrasing achieved.
**Conclusion**: this was never a fundamental limit of the Mamba3 recurrence -- it was a
training-data-design problem the whole time. The mechanism genuinely folds information past the
old window's limit once the training data actually teaches the specific association, not just
exposes the model to the raw fact.
