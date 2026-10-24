"""A/B the answer step on FIXED evidence: re-writes the answers of a saved v2 run
(same route plan, same graded chunks) with different answer settings, so any score
change comes from the answer step alone, not from retrieval / grading noise.

usage: python scripts/replay_answer.py <collection> <run.jsonl> <out.jsonl> [--words L,A,C] [--model M] [--rules NAME]
  --words  word limits for lookup,analysis,comparison (default: kb.query.LIMITS)
  --model  answer model (default: config.ANSWER_MODEL)
  --rules  extra answer rules: none | focused
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kb import llm, store  # noqa: E402
from kb.query import LIMITS, MIN_GRADE2, write_answer  # noqa: E402

RULES = {
    "none": "",
    "focused": ("- Every sentence must directly answer one checklist item; omit background, definitions and methodology the question "
                "did not ask for, and do not repeat a fact.\n"
                "- In tables, include only rows / columns for the checklist's dimensions."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("collection")
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--words", default="")
    ap.add_argument("--model", default=None)
    ap.add_argument("--rules", default="none", choices=list(RULES))
    a = ap.parse_args()
    words = dict(zip(["lookup", "analysis", "comparison"], map(int, a.words.split(",")))) if a.words else {}
    col = store.load(a.collection)
    runs = [json.loads(l) for l in open(a.run)]

    def redo(r):
        plan = next(s for s in r["trace"] if s["step_name"] == "route")["output"]
        qtype = plan.get("question_type") if plan.get("question_type") in LIMITS else "analysis"
        facets = [r["question"]] if qtype == "lookup" else ([f for f in plan.get("facets", []) if isinstance(f, str)] or [r["question"]])
        missing = [str(m) for m in plan.get("missing", []) if m]
        meta = [c.get("metadata", c) for c in r["chunks"]]  # contract format, or older runs
        g2 = [m["chunk_id"] for m in meta if m.get("grade") == 2]
        evidence = g2 if len(g2) >= MIN_GRADE2["answer"] else [m["chunk_id"] for m in meta]
        if not evidence:  # abstained in the original run: keep it
            return r
        usage = llm.Usage()
        ans = write_answer(col, r["question"], qtype, facets, missing, evidence, usage,
                           words=words.get(qtype), model=a.model, extra_rules=RULES[a.rules])
        return {**r, "answer": ans.strip(), "answer_usage": usage.to_dict()}

    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(redo, runs))
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} answers to {a.out}")


if __name__ == "__main__":
    main()
