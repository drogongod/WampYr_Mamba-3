#!/usr/bin/env python3
"""THE PROGRAMMING-AGENT TEST, 2026-10-14. Fairly scoped: not "can a 64K-parameter brick write
code from scratch" (nothing has ever claimed that) -- this tests whether the real vampire
mechanism (own-weights-first, donor-fallback, explicit accept, host-off retention-under-
interference) works for an ACTUAL programming task, same evidence-contract discipline as every
fact-retention test today. The donor's code is verified by ACTUALLY RUNNING IT, not eyeballed --
"a pretty answer is NOT proof" applies exactly as much to code as to trivia facts.
"""
import re
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"F:\my-agent-vampire-lucy")

import brickcluster_engine as E
import mamba3_lutgtlm_ingest as MI

E.LI = MI
E.WEIGHT_ARRAY_NAMES = ("Win",)
E.LI.STATE = os.path.join(os.path.dirname(__file__), "data", "checkpoints",
                          "programming_agent_test_mind.npz")
if os.path.exists(E.LI.STATE):
    os.remove(E.LI.STATE)

engine = E.BrickClusterEngine()

TASK = ("Write a Python function called reverse_string that takes a string argument and "
        "returns the string reversed. Reply with ONLY the function definition, no explanation, "
        "no markdown fences.")


def extract_code(text: str) -> str:
    # strip markdown fences if the donor added them despite the instruction not to
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


print("=== STEP 1: host-off, before any teaching ===")
candidate, error, matched = engine._lucy_candidate(TASK)
print(f"  matched={matched}")
assert not matched, "FAIL: claims to know this before anything was taught"
print("PASS: no brick matched -- she has never seen a programming task.\n")

print("=== STEP 2: ask the donor for real code (the real vampire fallback path) ===")
body = {"messages": [{"role": "user", "content": TASK}], "stream": False}
response = engine.complete(body)
route = response["vampire_route"]
donor_code_raw = response["choices"][0]["message"]["content"]
print(f"  donor raw response:\n{donor_code_raw}\n")

code = extract_code(donor_code_raw)
print(f"  extracted code:\n{code}\n")

print("=== STEP 3: VERIFY the donor's code by actually running it (not eyeballed) ===")
namespace = {}
try:
    exec(code, namespace)
    fn = namespace.get("reverse_string")
    assert fn is not None, "no reverse_string function defined"
    test_cases = [("hello", "olleh"), ("", ""), ("a", "a"), ("racecar", "racecar")]
    all_correct = True
    for inp, expected in test_cases:
        got = fn(inp)
        ok = got == expected
        all_correct = all_correct and ok
        print(f"  reverse_string({inp!r}) = {got!r}  expected={expected!r}  {'OK' if ok else 'WRONG'}")
    print(f"\n  REAL EXECUTION RESULT: {'ALL CORRECT' if all_correct else 'FAILED - code is wrong'}")
except Exception as exc:
    all_correct = False
    print(f"  REAL EXECUTION RESULT: CRASHED -- {type(exc).__name__}: {exc}")

if not all_correct:
    print("\nSTOPPING HERE -- donor code did not actually work, no point teaching/testing "
          "retention of something verified wrong. Report this as the real result.")
    sys.exit(0)

print("\n=== STEP 4: teach it (the real /accept path) ===")
result = engine.accept_experience(route["session_id"], route["experience_id"], action="accept")
update = result["weight_update"]
print(f"  brick_index={update.get('brick_index')} is_new_brick={update.get('is_new_brick')} "
      f"n_bricks={update.get('n_bricks')}")

print("\n=== STEP 5: host-off retest immediately after teaching ===")
candidate, error, matched = engine._lucy_candidate(TASK)
print(f"  matched={matched}")

print("\n=== STEP 6: teach an unrelated fact (interference) ===")
engine.learn_accepted("What is the capital of Japan?", "The capital of Japan is Tokyo.")

print("\n=== STEP 7: re-ask the programming task after unrelated learning ===")
candidate, error, matched_after = engine._lucy_candidate(TASK)
print(f"  matched={matched_after}")

print("\n=== RESULT ===")
print(f"  Donor code actually ran correctly (verified by execution): {all_correct}")
print(f"  Knew the task right after teaching:                       {matched}")
print(f"  Retained the task after unrelated interference:            {matched_after}")
print(f"  PROGRAMMING-AGENT EVIDENCE CONTRACT: "
      f"{'PASS' if (all_correct and matched and matched_after) else 'FAIL (reported honestly)'}")
