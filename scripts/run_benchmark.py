"""Run the query pipeline over benchmark questions and save outputs.

usage: python scripts/run_benchmark.py <collection> <out.jsonl> [--ids A01,B07] [--limit N] [--every K] [--offset O]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BENCH = Path(__file__).resolve().parent.parent / "Train" / "Benchmarks" / "benchmark_100.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("collection")
    ap.add_argument("out")
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="take every k-th question (spread across families)")
    ap.add_argument("--offset", type=int, default=0, help="start index for --every (a different sample of questions)")
    a = ap.parse_args()
    from kb.query import run_query

    rows = [json.loads(l) for l in BENCH.open()]
    if a.ids:
        want = set(a.ids.split(","))
        rows = [r for r in rows if r["id"] in want]
    rows = rows[a.offset :: a.every]
    if a.limit:
        rows = rows[: a.limit]

    done = set()
    out = Path(a.out)
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open()}
    with out.open("a") as f:
        for r in rows:
            if r["id"] in done:
                continue
            t = time.time()
            try:
                res = run_query(a.collection, r["question"])
            except Exception as e:  # keep going; record the failure
                print(f"{r['id']} FAILED: {e}", flush=True)
                continue
            gold_docs = set(r["benchmark_chunks_w_metadata"])
            got_docs = {c["document"] for c in res["chunks"]}
            rec = {"id": r["id"], "question": r["question"], "answer": res["answer"], "chunks": res["chunks"],
                   "usage": res["usage"], "latency_seconds": res["latency_seconds"], "step_seconds": res["step_seconds"],
                   "trace": res["trace"]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{r['id']} {time.time()-t:5.1f}s chunks={len(res['chunks']):3d} gold_chunks="
                  f"{sum(len(v) for v in r['benchmark_chunks_w_metadata'].values()):3d} "
                  f"docs got/gold={len(got_docs & gold_docs)}/{len(gold_docs)} extra_docs={len(got_docs - gold_docs)} "
                  f"tokens={res['usage']['prompt_tokens']}+{res['usage']['completion_tokens']}", flush=True)


if __name__ == "__main__":
    main()
