"""Ingestion: unzip -> parse (cached) -> document cards -> look-alike notes ->
contextual unit headers -> embeddings -> persisted collection."""
from __future__ import annotations

import hashlib
import json

import re
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH

from . import config, llm, store
import pymupdf

from .parse import PARSER_VERSION, PagePool, normalize_text, parse_pdf

CACHE_DIR = config.DATA_DIR.parent / "cache" / "parsed"

CARD_SYSTEM = (
    "You catalogue documents for a financial research knowledge base. "
    "Read the excerpt of one document and describe it precisely. Respond only with JSON."
)
CARD_PROMPT = """File name: {filename}
Pages: {n_pages}
Outline (first headings): {outline}

Opening pages:
{opening}

Sampled later pages:
{sample}

Return JSON with these keys:
- "title": the document's own full title
- "short_title": <= 10 words that uniquely identify it, including the entity and the period/version (e.g. "Refex Industries Annual Report FY2025-26")
- "entity": the company / institution the document is about (or the authors for a paper)
- "issuer": who published it
- "doc_type": one of annual report, integrated annual report, regulation, final rule, proposed rule, supervisory guidance, FAQ, accounting standard, implementation guide, handbook, board memo, fact sheet, research paper, technical document, other
- "period": reporting period / fiscal year covered, or null
- "date": issue / publication date as written, or null
- "jurisdiction": country or body (e.g. US, India, Saudi Arabia, UAE, Basel Committee/international), or null
- "status": final | proposed | consultation | draft | n/a
- "topics": 8-15 specific topic keywords
- "summary": 2-3 sentences on what the document covers and what questions it can answer"""

SIBLING_SYSTEM = "You compare two similar documents in a knowledge base. Respond only with JSON."
SIBLING_PROMPT = """Two documents in the same collection look alike. Explain how to tell them apart when deciding which one answers a question.

Document A ({a_id}): {a_card}
Opening of A: {a_text}

Document B ({b_id}): {b_card}
Opening of B: {b_text}

Return JSON: {{"difference": "<one or two sentences: what differs (period, version, scope, status, entity) and when to use A vs B>"}}"""


def _sha(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_cached(path: Path, doc_id: str, pool) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # cache key includes parser + PyMuPDF versions: output changes across versions (2.0 drops spaces)
    cp = CACHE_DIR / f"{_sha(path)}-p{PARSER_VERSION}-m{pymupdf.VersionBind}.json"
    if cp.exists():
        d = json.loads(cp.read_text())
    else:
        d = parse_pdf(str(path), "DOC", path.name, pool=pool).to_dict()
        cp.write_text(json.dumps(d))
    # re-key cached ids to this collection's doc id
    raw = json.dumps(d).replace('"DOC:', f'"{doc_id}:').replace('"doc_id": "DOC"', f'"doc_id": "{doc_id}"')
    d = json.loads(raw)
    d["filename"] = path.name
    for u in d["units"]:  # idempotent; also upgrades caches written before normalisation existed
        u["text"] = normalize_text(u["text"])
    for s in d["sections"]:
        s["title"] = normalize_text(s["title"])
        s["path"] = [normalize_text(x) for x in s["path"]]
    return d


def _page_texts(doc: dict) -> dict[int, str]:
    pages: dict[int, list[str]] = {}
    for u in doc["units"]:
        pages.setdefault(u["page"], []).append(u["text"])
    return {p: "\n".join(t) for p, t in pages.items()}


def _card_job(doc: dict, usage: llm.Usage) -> dict:
    pt = _page_texts(doc)
    pages = sorted(pt)
    opening, budget = [], 7000
    for p in pages:
        if budget <= 0:
            break
        t = pt[p][:budget]
        opening.append(f"[p{p}] {t}")
        budget -= len(t)
    mid = [p for p in pages if p > pages[0] + 3]
    sample = []
    for p in (mid[len(mid) // 3], mid[2 * len(mid) // 3]) if len(mid) >= 3 else []:
        sample.append(f"[p{p}] {pt[p][:1500]}")
    outline = [s["title"] for s in doc["sections"] if 0 < s["level"] <= 2][:50]
    return dict(
        system=CARD_SYSTEM,
        user=CARD_PROMPT.format(
            filename=doc["filename"], n_pages=doc["n_pages"], outline="; ".join(outline)[:2500],
            opening="\n".join(opening), sample="\n".join(sample) or "(none)",
        ),
        max_tokens=4000,
        usage=usage,
        name="doc_card",
    )


def _minhash(doc: dict) -> MinHash:
    words = store.tokenize(" ".join(u["text"] for u in doc["units"][:4000]))
    m = MinHash(num_perm=128)
    for i in range(0, max(0, len(words) - 5), 2):
        m.update(" ".join(words[i : i + 5]).encode())
    return m


def _fallback_card(doc: dict) -> dict:
    stem = Path(doc["filename"]).stem
    return {"title": stem, "short_title": stem, "entity": None, "issuer": None, "doc_type": "other", "period": None,
            "date": None, "jurisdiction": None, "status": "n/a", "topics": [], "summary": ""}


def _embed_cached(texts: list[str]) -> np.ndarray:
    """Embed with an on-disk cache keyed by (model, text) so rebuilds only embed changed units."""
    cdir = config.DATA_DIR.parent / "cache" / "emb"
    cdir.mkdir(parents=True, exist_ok=True)
    cfile = cdir / (re.sub(r"[^A-Za-z0-9]", "_", config.EMBED_MODEL) + ".npz")
    cache: dict[str, np.ndarray] = {}
    if cfile.exists():
        z = np.load(cfile)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    todo = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items())
    if todo:
        new = store.embed([t for _, t in todo])
        cache.update({k: v for (k, _), v in zip(todo, new)})
        np.savez(cfile, keys=np.array(list(cache)), vecs=np.stack(list(cache.values())))
    return np.stack([cache[k] for k in keys])


def build_collection(name: str, pdf_paths: list[Path], log=print) -> dict:
    t0 = time.time()
    usage = llm.Usage()
    out = store.collection_dir(name)
    tmp = out.with_name(out.name + ".building")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    # 1. parse
    docs = []
    with PagePool(config.WORKERS) as pool:
        for i, p in enumerate(sorted(pdf_paths, key=lambda x: x.name.lower())):
            t = time.time()
            d = _parse_cached(p, f"d{i:03d}", pool)
            docs.append(d)
            log(f"parsed {p.name}: {d['n_pages']}p, {len(d['units'])} units, {len(d['ocr_pages'])} OCR pages ({time.time()-t:.1f}s)")
    t_parse = time.time() - t0

    # 2. document cards
    cards = llm.chat_many([_card_job(d, usage) for d in docs], concurrency=config.LLM_CONCURRENCY)
    n_failed = sum(1 for c in cards if not isinstance(c, dict))
    if docs and n_failed > len(docs) / 2:
        # routing depends on the cards: refuse to publish a collection built on placeholders
        # (e.g. the LLM provider is down or the API budget is exhausted)
        raise RuntimeError(f"document card generation failed for {n_failed}/{len(docs)} documents; "
                           "check the LLM API key / budget and re-run /index")
    for d, c in zip(docs, cards):
        c = c if isinstance(c, dict) else _fallback_card(d)
        c.update(doc_id=d["doc_id"], filename=d["filename"], n_pages=d["n_pages"], ocr_pages=len(d["ocr_pages"]), siblings=[])
        d["card"] = c
    log(f"cards done ({usage.calls} LLM calls)")

    # 3. look-alike detection: text overlap (MinHash) or same entity + doc type
    mh = {d["doc_id"]: _minhash(d) for d in docs}
    lsh = MinHashLSH(threshold=0.15, num_perm=128)
    for k, m in mh.items():
        lsh.insert(k, m)
    pairs = set()
    for k, m in mh.items():
        for o in lsh.query(m):
            if o != k:
                pairs.add(tuple(sorted((k, o))))
    by_key: dict[tuple, list[str]] = {}
    for d in docs:
        c = d["card"]
        ent = re.sub(r"[^a-z0-9]", "", (c.get("entity") or "").lower())[:20]
        if ent:
            by_key.setdefault((ent, (c.get("doc_type") or "").lower()), []).append(d["doc_id"])
    for ids in by_key.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add((ids[i], ids[j]))
    dmap = {d["doc_id"]: d for d in docs}

    def brief(c):
        return json.dumps({k: c.get(k) for k in ("title", "entity", "doc_type", "period", "date", "status", "jurisdiction")})

    def opening(d):
        pt = _page_texts(d)
        return " ".join(pt[p] for p in sorted(pt)[:2])[:1500]

    pairs = sorted(pairs)
    notes = llm.chat_many([
        dict(system=SIBLING_SYSTEM, user=SIBLING_PROMPT.format(
            a_id=a, a_card=brief(dmap[a]["card"]), a_text=opening(dmap[a]),
            b_id=b, b_card=brief(dmap[b]["card"]), b_text=opening(dmap[b])),
            max_tokens=2000, usage=usage, name="sibling_note")
        for a, b in pairs
    ])
    for (a, b), n in zip(pairs, notes):
        diff = (n or {}).get("difference", "") if isinstance(n, dict) else ""
        dmap[a]["card"]["siblings"].append({"doc_id": b, "jaccard": round(mh[a].jaccard(mh[b]), 2), "difference": diff})
        dmap[b]["card"]["siblings"].append({"doc_id": a, "jaccard": round(mh[a].jaccard(mh[b]), 2), "difference": diff})
    log(f"look-alike pairs: {len(pairs)}")

    # 4. units with contextual headers, sections
    units, sections = [], []
    for d in docs:
        c = d["card"]
        sec_path = {s["id"]: s["path"] for s in d["sections"]}
        for s in d["sections"]:
            sections.append(s)
        for u in d["units"]:
            path = " > ".join(sec_path.get(u["section_id"], [])[1:][-3:])
            header = f"[{c.get('short_title') or d['filename']} | {path or 'front matter'} | p{u['page']}]"
            u["header"] = header
            u["search_text"] = header + "\n" + u["text"]
            units.append(u)

    # 5. embeddings
    t = time.time()
    vecs = _embed_cached([u["search_text"][:1200] for u in units])  # model reads ~256 tokens anyway
    log(f"embedded {len(units)} units ({time.time()-t:.1f}s)")

    (tmp / "cards.json").write_text(json.dumps([d["card"] for d in docs], indent=1))
    (tmp / "sections.json").write_text(json.dumps(sections))
    with (tmp / "units.jsonl").open("w") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")
    np.save(tmp / "vectors.npy", vecs)
    stats = {
        "collection": name, "documents": len(docs), "pages": sum(d["n_pages"] for d in docs),
        "units": len(units), "sections": len(sections), "lookalike_pairs": len(pairs),
        "parse_seconds": round(t_parse, 1), "total_seconds": round(time.time() - t0, 1), **usage.to_dict(),
    }
    (tmp / "meta.json").write_text(json.dumps(stats, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    store.invalidate(name)
    log(f"collection '{name}' ready: {stats}")
    return stats


def index_zip(name: str, zip_path: Path, log=print) -> dict:
    pdfs, skipped = extract_zip(name, zip_path, log=log)
    stats = build_collection(name, pdfs, log=log)
    stats["skipped_non_pdf"] = skipped
    return stats


def extract_zip(name: str, zip_path: Path, log=print) -> tuple[list[Path], list[str]]:
    """Unpack the PDFs of an upload into the collection's raw folder (fast; raises
    zipfile.BadZipFile / ValueError for unusable uploads before any heavy work starts)."""
    raw = store.collection_dir(name).with_name(store.collection_dir(name).name + ".raw")
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)
    skipped: list[str] = []
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            if m.is_dir() or "__MACOSX" in m.filename or Path(m.filename).name.startswith("."):
                continue
            if not m.filename.lower().endswith(".pdf"):
                skipped.append(m.filename)  # only page-trackable PDFs are indexed
                continue
            target = raw / Path(m.filename).name  # flatten; guards against path traversal
            k = 1
            while target.exists():  # same file name in two folders
                target = raw / f"{Path(m.filename).stem}__{k}.pdf"
                k += 1
            with z.open(m) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    pdfs = sorted(raw.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"the zip contains no PDF files (skipped: {skipped[:20]})")
    if skipped:
        log(f"skipped {len(skipped)} non-PDF files: {skipped[:20]}")
    return pdfs, skipped
