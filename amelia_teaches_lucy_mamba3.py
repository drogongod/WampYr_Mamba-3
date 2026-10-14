#!/usr/bin/env python3
"""Mamba3-quinary version of F:\\my-agent-vampire-lucy\\amelia_teaches_lucy.py -- identical
teacher loop (same Amelia-voiced donor, same category rotation, same skip/truncation checks,
same periodic sleep() discipline), monkeypatching brickcluster_engine's LI module to the
Mamba3 ingest/cluster instead of copy-pasting the whole proven engine.
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

# Monkeypatch: every E.LI.xxx call inside the proven engine now resolves against the Mamba3
# ingest module instead of the original lutgtlm_ingest -- reuses ALL of BrickClusterEngine's
# logic (complete/accept_experience/sleep/_lucy_candidate) completely unmodified.
E.LI = MI
# Compatibility shim: learn_accepted's before/after diff expects FixedStepBrick's named
# numpy level-arrays; Mamba3QuinaryBrickLM exposes the same concept under a single ".Win"
# property (all its quinary levels, concatenated) -- see mamba3_brick_lm.py's docstring.
E.WEIGHT_ARRAY_NAMES = ("Win",)

N_FACTS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SLEEP_EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 100
CHECKPOINT = os.path.join(os.path.dirname(__file__), "data", "checkpoints",
                          "amelia_teaches_lucy_mamba3_mind.npz")
E.LI.STATE = CHECKPOINT

IDENTITY_PATH = r"C:\Users\Azazel\Desktop\Amelia's Dataset\Identity_Amelia_short.txt"
AMELIA_IDENTITY = open(IDENTITY_PATH, encoding="utf-8").read().strip()

TEACHER_SYSTEM = (
    AMELIA_IDENTITY + "\n\n"
    "You are quizzing a student named Lucy. Your ONLY job right now: ask exactly ONE short, "
    "clear, factual question she could learn from. Never repeat a question already asked (a "
    "list of already-asked questions follows). Reply with ONLY the question text, nothing else "
    "-- no preamble, no commentary, no quotation marks."
)

CATEGORIES = [
    "geography", "history", "physics", "chemistry", "biology", "mathematics/arithmetic",
    "literature", "movies", "music", "video games", "mythology and folklore", "sports",
    "art and architecture", "technology and computing", "food and cooking",
    "animals and nature", "space and astronomy", "famous historical figures",
    "language and etymology", "inventions and discoveries",
]


def category_for(fact_index: int) -> str:
    return CATEGORIES[fact_index % len(CATEGORIES)]


AVOID_LIST_WINDOW = 300


def build_teacher_body(asked_already: list[str], category: str) -> dict:
    avoid = "\n".join(f"- {q}" for q in asked_already[-AVOID_LIST_WINDOW:]) or "(none yet)"
    return {
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": f"Already asked, do not repeat any of these:\n{avoid}\n\n"
                                         f"Your question for THIS round must be about: {category}. "
                                         "Pick ONE specific fact in that category and ask about it "
                                         "directly -- the first reasonable fact that comes to mind "
                                         "is fine, do not deliberate over whether it's famous "
                                         "enough. Ask your next question now."},
        ],
        "model": "vampire-lucywest", "stream": False, "temperature": 0.9, "max_tokens": 600,
    }


def ask_amelia_for_question(asked_already: list[str], category: str) -> str:
    body = build_teacher_body(asked_already, category)
    response = E.post_json(E.REFERENCE_URL.rstrip("/") + "/v1/chat/completions", body, timeout=120)
    text = response["choices"][0]["message"].get("content") or ""
    return text.strip().strip('"').strip()


def canonical(q: str) -> str:
    return " ".join(q.lower().split())


def main():
    engine = E.BrickClusterEngine()
    print(f"[start] n_bricks={len(engine.cluster.bricks)} seen={engine.seen}", flush=True)
    print(f"[plan] {N_FACTS} facts, auto-accept, sleep() every {SLEEP_EVERY} facts, "
          f"WAMPYR_MAMBA3 substrate\n", flush=True)

    asked_text: list[str] = []
    asked_keys: set[str] = set()
    taught: list[tuple[str, str]] = []
    log_path = os.path.join(os.path.dirname(__file__), "data", "experience",
                             "amelia_teaching_log_mamba3.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    i = 0
    attempts = 0
    while i < N_FACTS and attempts < N_FACTS * 3:
        attempts += 1
        category = category_for(i)
        try:
            question = ask_amelia_for_question(asked_text, category)
        except Exception as exc:
            print(f"  [teacher error] {type(exc).__name__}: {exc}, retrying", flush=True)
            continue
        key = canonical(question)
        truncated = bool(question) and question.rstrip()[-1:] not in "?."
        if not question or key in asked_keys or len(question) > 300 or truncated:
            reason = "truncated" if truncated else "duplicate/empty/too-long"
            print(f"  [skip {reason}] {question!r}", flush=True)
            continue
        asked_keys.add(key)
        asked_text.append(question)

        body = {"messages": [{"role": "user", "content": question}], "stream": False}
        try:
            response = engine.complete(body)
        except Exception as exc:
            print(f"  [complete error on {question!r}] {type(exc).__name__}: {exc}", flush=True)
            continue
        route = response["vampire_route"]
        donor_answer = response["choices"][0]["message"]["content"]
        try:
            result = engine.accept_experience(route["session_id"], route["experience_id"],
                                               action="accept")
        except Exception as exc:
            print(f"  [accept error] {type(exc).__name__}: {exc}", flush=True)
            continue
        update = result["weight_update"]
        i += 1
        print(f"[{i}/{N_FACTS}] ({category}) Q: {question}", flush=True)
        print(f"       A: {donor_answer}", flush=True)
        print(f"       brick={update.get('brick_index')} new={update.get('is_new_brick')} "
              f"changed_levels={update.get('changed_levels')} n_bricks={update.get('n_bricks')}",
              flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"question": question, "answer": donor_answer,
                                 "auto_accepted": True, "weight_update": update,
                                 "at_ns": time.time_ns()}, ensure_ascii=False) + "\n")
        taught.append((question, donor_answer))

        if i % SLEEP_EVERY == 0:
            t0 = time.time()
            sleep_result = engine.sleep()
            print(f"       [periodic sleep @ {i}] {sleep_result} ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n[before sleep] n_bricks={len(engine.cluster.bricks)}", flush=True)
    sleep_result = engine.sleep()
    print(f"[sleep] {sleep_result}", flush=True)

    print("\n=== retention check: host-off, all taught questions ===", flush=True)
    matched_count = 0
    for q, a in taught:
        candidate, error, matched = engine._lucy_candidate(q)
        matched_count += int(matched)
        print(f"  matched={matched!s:5}  {q!r}", flush=True)
    print(f"\n[final] {matched_count}/{len(taught)} matched host-off; "
          f"n_bricks={len(engine.cluster.bricks)} seen={engine.seen}", flush=True)


if __name__ == "__main__":
    main()
