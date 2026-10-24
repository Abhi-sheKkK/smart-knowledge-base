"""How much of each gold chunk's text does our parser recover on the same page?

Token recall = |gold tokens found in our page text| / |gold tokens|, averaged per document.
Low values point at parsing / OCR losses before any retrieval happens.
"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pymupdf  # noqa: E402
from kb.index import CACHE_DIR, _sha  # noqa: E402
from kb.parse import PARSER_VERSION  # noqa: E402
from kb.parse import normalize_text  # noqa: E402

ALIASES = {  # benchmark file names that differ from the folder (see APPROACH.md §2.3)
    "fact-sheet-20260319.pdf": "fact-sheet-SUMMARY.pdf",
    "Guidelines on Management of Problem Loans.pdf": "SAMA_EN_3575_VER1.pdf",
}
TOK = re.compile(r"[a-z]{3,}|\d+(?:[.,]\d+)*")


def toks(s):
    return set(TOK.findall(normalize_text(s).lower()))


def main():
    files = {p.name: p for d in ("Train/Documents", "Train/Downloaded_Missing") for p in (ROOT / d).glob("*.pdf")}
    pages_cache = {}

    def page_text(name):
        if name not in pages_cache:
            d = json.loads((CACHE_DIR / f"{_sha(files[name])}-p{PARSER_VERSION}-m{pymupdf.VersionBind}.json").read_text())
            pt = defaultdict(str)
            for u in d["units"]:
                pt[u["page"]] += " " + normalize_text(u["text"])
            for s in d["sections"]:  # headings are kept as section titles, not unit text
                pt[s["page_start"]] += " " + normalize_text(s["title"])
            pages_cache[name] = {p: toks(t) for p, t in pt.items()}
        return pages_cache[name]

    per_doc = defaultdict(list)
    worst = []
    for line in (ROOT / "Train/Benchmarks/benchmark_100.jsonl").open():
        r = json.loads(line)
        for doc, chunks in r["benchmark_chunks_w_metadata"].items():
            name = ALIASES.get(doc, doc)
            if name not in files:
                continue
            pt = page_text(name)
            for c in chunks:
                g = toks(c["content"])
                if len(g) < 8 or c["section_name"].startswith("math_"):
                    continue
                pg = c["metadata"]["page_numbers"][0]
                rec = len(g & pt.get(pg, set())) / len(g)
                per_doc[name].append(rec)
                if rec < 0.5:
                    worst.append((rec, name, pg, c["content"][:120].replace("\n", " ")))
    allv = [v for vs in per_doc.values() for v in vs]
    print(f"overall same-page token recall: {sum(allv)/len(allv):.3f} over {len(allv)} gold chunks")
    for name, vs in sorted(per_doc.items(), key=lambda x: sum(x[1]) / len(x[1])):
        low = sum(v < 0.5 for v in vs)
        print(f"  {sum(vs)/len(vs):.3f}  n={len(vs):4d}  <0.5: {low:3d}  {name}")
    print("\nworst examples:")
    for w in sorted(worst)[:12]:
        print(f"  {w[0]:.2f} {w[1][:35]} p{w[2]}: {w[3]}")


if __name__ == "__main__":
    main()
