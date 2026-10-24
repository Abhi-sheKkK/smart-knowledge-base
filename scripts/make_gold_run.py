"""Judge-ceiling test input: a run file whose answers and chunks ARE the gold ones.

Scoring it shows the best the judge can give (1.0 on every metric if the judge were
perfect); anything below is judge noise, not system error.

usage: python scripts/make_gold_run.py <out.jsonl> [--every K]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "Train" / "Benchmarks" / "benchmark_100.jsonl"


def main():
    out = Path(sys.argv[1])
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1
    rows = [json.loads(l) for l in BENCH.open()][::every]
    with out.open("w") as f:
        for r in rows:
            chunks = [{"text": c["content"], "document": d, "page": c["metadata"]["page_numbers"][0]}
                      for d, cs in r["benchmark_chunks_w_metadata"].items() for c in cs]
            f.write(json.dumps({"id": r["id"], "question": r["question"], "answer": r["benchmark_answer"],
                                "chunks": chunks, "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                                "latency_seconds": 0}) + "\n")
    print(f"wrote {len(rows)} gold rows to {out}")


if __name__ == "__main__":
    main()
