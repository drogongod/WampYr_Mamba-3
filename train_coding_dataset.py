#!/usr/bin/env python3
"""Teach Lucy (WampYr_Mamba-3 substrate) directly from coding_dataset_final.jsonl's own real
(instruction, output) pairs -- no donor round-trip needed, the dataset already has verified-
real answers (sampled and inspected 2026-10-15: genuinely diverse, real problems, not the
templated-duplicate filler found in Vibe-Coding-Instruct.jsonl).

Handles the one real schema inconsistency found while sampling: most rows use "output", a few
use "response" instead.
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"F:\my-agent-vampire-lucy")

import brickcluster_engine as E
import mamba3_lutgtlm_ingest as MI

E.LI = MI
E.WEIGHT_ARRAY_NAMES = ("Win",)

N_FACTS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SLEEP_EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 100
DATASET_PATH = r"D:\Datasets\new\coding\coding_dataset_final.jsonl"

CHECKPOINT = os.path.join(os.path.dirname(__file__), "data", "checkpoints",
                          "coding_dataset_mind.npz")
E.LI.STATE = CHECKPOINT

MAX_PROMPT_CHARS = 2000   # defensive cap -- some entries could be very long; this project's own
# learn_accepted already caps at [-4096:]/[:2048] internally, but capping here too avoids
# wasting time constructing absurdly long batches for outlier rows.
MAX_ANSWER_CHARS = 2000


def iter_examples(path, limit):
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if count >= limit:
                return
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            instruction = row.get("instruction", "").strip()
            answer = (row.get("output") or row.get("response") or "").strip()
            if not instruction or not answer:
                continue
            if len(instruction) > MAX_PROMPT_CHARS or len(answer) > MAX_ANSWER_CHARS:
                continue
            count += 1
            yield instruction, answer


def main():
    engine = E.BrickClusterEngine()
    print(f"[start] n_bricks={len(engine.cluster.bricks)} seen={engine.seen}", flush=True)
    print(f"[plan] {N_FACTS} real coding examples from coding_dataset_final.jsonl, "
          f"sleep() every {SLEEP_EVERY}\n", flush=True)

    log_path = os.path.join(os.path.dirname(__file__), "data", "experience",
                             "coding_dataset_teaching_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    taught = []
    i = 0
    for instruction, answer in iter_examples(DATASET_PATH, N_FACTS):
        try:
            update = engine.learn_accepted(instruction, answer)
        except Exception as exc:
            print(f"  [learn error] {type(exc).__name__}: {exc}", flush=True)
            continue
        i += 1
        print(f"[{i}/{N_FACTS}] Q: {instruction[:100]!r}", flush=True)
        print(f"       A: {answer[:100]!r}", flush=True)
        print(f"       status={update.get('status')} brick={update.get('brick_index')} "
              f"new={update.get('is_new_brick')} n_bricks={update.get('n_bricks')}", flush=True)
        taught.append((instruction, answer))

        if i % SLEEP_EVERY == 0:
            t0 = time.time()
            sleep_result = engine.sleep()
            print(f"       [periodic sleep @ {i}] {sleep_result} ({time.time()-t0:.1f}s)", flush=True)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"instruction": instruction, "answer": answer,
                                 "weight_update": update, "at_ns": time.time_ns()},
                                ensure_ascii=False) + "\n")

    print(f"\n[before final sleep] n_bricks={len(engine.cluster.bricks)}", flush=True)
    sleep_result = engine.sleep()
    print(f"[sleep] {sleep_result}", flush=True)

    print("\n=== retention check: host-off, all taught examples ===", flush=True)
    matched_count = 0
    for q, a in taught:
        candidate, error, matched = engine._lucy_candidate(q)
        matched_count += int(matched)
        print(f"  matched={matched!s:5}  {q[:80]!r}", flush=True)
    print(f"\n[final] {matched_count}/{len(taught)} matched host-off; "
          f"n_bricks={len(engine.cluster.bricks)} seen={engine.seen}", flush=True)


if __name__ == "__main__":
    main()
