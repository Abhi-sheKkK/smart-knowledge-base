"""Per-collection persistence and in-memory search (dense + BM25, scoped to documents)."""
from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOP = set("the a an of and or to in on for by with as at is are was were be been this that these those it its from which what how does do".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@lru_cache(maxsize=1)
def embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(config.EMBED_MODEL)


def embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    vecs = np.array(list(embedder().embed(texts, batch_size=batch_size)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


def embed_query(text: str) -> np.ndarray:
    if "bge" in config.EMBED_MODEL.lower():  # bge models expect an instruction prefix on queries
        text = "Represent this sentence for searching relevant passages: " + text
    return embed([text])[0]


def collection_dir(name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return config.DATA_DIR / safe


class Collection:
    """Loaded collection: docs (cards), sections, units, vectors, BM25."""

    def __init__(self, name: str):
        d = collection_dir(name)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"collection '{name}' is not indexed")
        self.name = name
        self.meta = json.loads((d / "meta.json").read_text())
        self.docs: dict[str, dict] = {c["doc_id"]: c for c in json.loads((d / "cards.json").read_text())}
        self.sections: dict[str, dict] = {s["id"]: s for s in json.loads((d / "sections.json").read_text())}
        self.units: list[dict] = [json.loads(l) for l in (d / "units.jsonl").open()]
        self.unit_idx = {u["id"]: i for i, u in enumerate(self.units)}
        self.vecs = np.load(d / "vectors.npy")
        self.bm25 = BM25Okapi([tokenize(u["search_text"]) for u in self.units])
        self.doc_rows: dict[str, np.ndarray] = {}
        for i, u in enumerate(self.units):
            self.doc_rows.setdefault(u["doc_id"], []).append(i)
        self.doc_rows = {k: np.array(v) for k, v in self.doc_rows.items()}
        self.children: dict[str, list[str]] = {}
        for s in self.sections.values():
            if s["parent"]:
                self.children.setdefault(s["parent"], []).append(s["id"])

    # ------------------------------------------------------------ search
    def search(self, queries: list[str], doc_ids: list[str] | None, k: int = 40) -> list[tuple[int, float]]:
        """Reciprocal-rank fusion of dense + BM25 over the units of `doc_ids` (all if None)."""
        rows = np.concatenate([self.doc_rows[d] for d in doc_ids if d in self.doc_rows]) if doc_ids else np.arange(len(self.units))
        if len(rows) == 0:
            return []
        fused: dict[int, float] = {}
        for q in queries:
            dense = self.vecs[rows] @ embed_query(q)
            sparse = np.asarray(self.bm25.get_scores(tokenize(q)))[rows]
            for scores in (dense, sparse):
                top = np.argsort(-scores)[: k * 2]
                for rank, j in enumerate(top):
                    if scores[j] <= 0:
                        continue
                    r = int(rows[j])
                    fused[r] = fused.get(r, 0.0) + 1.0 / (60 + rank)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]

    def section_units(self, sid: str) -> list[str]:
        """All unit ids under a section, including descendants, in document order."""
        out, stack = [], [sid]
        while stack:
            s = stack.pop()
            out.extend(self.sections[s]["unit_ids"])
            stack.extend(reversed(self.children.get(s, [])))
        return sorted(out, key=lambda u: self.unit_idx[u])


_cache: dict[str, Collection] = {}
_lock = threading.Lock()


def load(name: str, reload: bool = False) -> Collection:
    with _lock:
        if reload or name not in _cache:
            _cache[name] = Collection(name)
        return _cache[name]


def invalidate(name: str):
    with _lock:
        _cache.pop(name, None)
