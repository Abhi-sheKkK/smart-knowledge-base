"""Context precision / recall exactly as the organisers' API contract scores them (no LLM):
a returned chunk is relevant if its document matches a benchmark document and its pages
overlap that document's benchmark pages; recall = share of benchmark (document, page) pairs
covered. Document names match ignoring case, extension and punctuation.

usage: python scripts/page_metrics.py <run.jsonl> [<run.jsonl> ...]
"""
import json
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "Train" / "Benchmarks" / "benchmark_100.jsonl"
# benchmark names that differ from the folder for the same document (see APPROACH.md)
ALIASES = {"fact-sheet-20260319": "fact-sheet-summary", "guidelines on management of problem loans": "sama_en_3575_ver1"}


def norm(name: str) -> str:
    stem = re.sub(r"\.pdf$", "", name.strip().lower())
    stem = ALIASES.get(stem, stem)
    return re.sub(r"[^a-z0-9]", "", stem)


def chunk_doc_pages(c: dict) -> tuple[str, set[int]]:
    if "metadata" in c:  # contract format
        return norm(c["metadata"]["file_name"]), set(c["metadata"]["page_numbers"])
    return norm(c["document"]), {c["page"]}  # earlier run format


def main():
    gold = {}
    for l in BENCH.open():
        r = json.loads(l)
        pairs = {(norm(d), p) for d, cs in r["benchmark_chunks_w_metadata"].items() for c in cs for p in c["metadata"]["page_numbers"]}
        gold[r["id"]] = pairs
    for path in sys.argv[1:]:
        ps, rs = [], []
        for l in open(path):
            r = json.loads(l)
            g = gold[r["id"]]
            chunks = [chunk_doc_pages(c) for c in r["chunks"]]
            if not g:  # nothing to retrieve: perfect only if nothing was returned
                ps.append(1.0 if not chunks else 0.0); rs.append(1.0 if not chunks else 0.0); continue
            rel = [any((d, p) in g for p in pages) for d, pages in chunks]
            covered = {(d, p) for d, pages in chunks for p in pages if (d, p) in g}
            ps.append(sum(rel) / len(chunks) if chunks else 0.0)
            rs.append(len(covered) / len(g))
        print(f"{path}: context precision {sum(ps)/len(ps):.3f} · context recall {sum(rs)/len(rs):.3f} · n={len(ps)}")


if __name__ == "__main__":
    main()
