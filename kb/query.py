"""Query pipeline: route -> retrieve -> grade -> answer.

1. route     (1 LLM call)  read the auto-built document catalogue; pick documents, question
                           type, checklist and search queries; flag documents that are missing
2. retrieve  (no LLM)      dense + BM25 search inside the routed documents, one search per
                           query / checklist item; hits expanded to their small leaf section;
                           near-duplicates collapsed; per-document round-robin budget
3. grade     (parallel)    every candidate graded 0 / 1 / 2 for this question
4. answer    (1 LLM call)  written only from grade-2 passages, bounded length, cited;
                           abstains when nothing is directly relevant

Every step appends {step_name, description, input, output} to the trace and is a Langfuse span."""
from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager, nullcontext

from . import config, llm, store

# per question type: search hits kept, candidates graded, leaf-section expansion size, answer words
LIMITS = {
    "lookup": {"hits": 30, "candidates": 60, "leaf": 4, "words": 150},
    "analysis": {"hits": 150, "candidates": 200, "leaf": 10, "words": 600},
    "comparison": {"hits": 180, "candidates": 240, "leaf": 10, "words": 650},
}
GRADE_BATCH = 10
GRADE_PASSAGE_CHARS = 1800
GRADE_CONCURRENCY = 24
MIN_GRADE2 = {"answer": 8, "chunks": 15}  # fall back to grade-1 context below these counts
MAX_EVIDENCE_CHARS = 120_000

ROUTE_SYSTEM = (
    "You are the planner of a financial research knowledge base. You read a user question and a catalogue "
    "of the documents in the collection, and decide which documents can answer it. Respond only with JSON."
)
ROUTE_PROMPT = """Question:
{question}

Document catalogue:
{catalogue}

Decide:
1. "question_type": "lookup" (one specific fact or a short list), "analysis" (explain / assess within one subject), or "comparison" (across entities, documents, regimes or periods).
2. "facets": the distinct things the answer must cover (1 for a lookup; 3-8 otherwise), as short checklist items.
3. "documents": every document needed, as {{"doc_id": ..., "why": ...}}. Match entity, period / fiscal year, version and status exactly; when look-alike documents exist (see "siblings") pick the one the question refers to. Prefer a specific document (named reform, proposal, year, version) over a large consolidated compendium unless the specific one lacks something needed. Include standards / guidance only when the answer must apply or explain them.
4. "missing": documents the question refers to that are NOT in the catalogue. Empty list if none.
5. "search_queries": 3-6 short keyword queries in the documents' own vocabulary, covering all facets.
6. "confidence": "high" if the right documents are clear, "low" otherwise.

Return JSON with exactly those keys."""

GRADE_SYSTEM = "You are a strict evidence screener for a financial research assistant. Respond only with JSON."
GRADE_PROMPT = """Question:
{question}

Checklist:
{facets}

Documents in scope: {scope}

Passages:
{passages}

{strictness}Grade every passage for THIS question. The test for grade 2 is: is the passage genuinely relevant and useful for answering the question?
- 2 = directly answers a checklist item (states the facts, figures, rules or findings the answer is built from)
- 1 = context the answer needs to be correct or complete (a definition, condition, comparison period or caveat for a grade-2 fact)
- 0 = not needed: only shares the topic, a table of contents, boilerplate, or the wrong entity / period / document version

Return JSON: {{"results": [{{"id": "<passage id>", "grade": 0|1|2}}, ...]}} with one entry per passage."""

LOOKUP_STRICTNESS = ("This question asks for one specific fact: grade 2 ONLY passages that state that fact (or a qualifier "
                     "the answer needs). Background on the same topic is grade 0.\n\n")

ANSWER_SYSTEM = (
    "You are a meticulous financial research analyst. You answer strictly from the evidence provided, "
    "cite every claim, and never add facts that are not in the evidence."
)
ANSWER_PROMPT = """Question:
{question}

Checklist the answer must cover:
{facets}

{missing}Evidence passages (each starts with [E#] and its source):
{evidence}

Write the answer in markdown.
Rules:
- Use ONLY the evidence. Every factual sentence or bullet ends with a citation in the form [<file name> p<page>]; several pages may be combined like [<file name> p12; p14].
- {coverage}
- If the evidence does not contain something the question asks for, say so plainly in one sentence ("The documents do not state ..."). Never guess, never use outside knowledge, no generic commentary.
- Keep entities, periods and document versions distinct; label figures from different years or documents.
- {style}"""

STYLE = {
    "lookup": "Give the direct answer first (bold the key figure / name), then the closely related specifics stated in the same passages (e.g. the full schedule, the comparison period, the condition it applies under) and any caveat about what the evidence does not confirm. At most {words} words.",
    "analysis": "Start with a 2-3 sentence bottom line, then one short section per checklist item with bullets or a table where figures are compared. At most {words} words: pick the facts that matter most; summarise rather than listing every figure.",
    "comparison": "Start with a 2-3 sentence bottom line, then compare item by item (a table is welcome) and state the key differences given in the evidence. At most {words} words: pick the facts that matter most; summarise rather than listing every figure.",
}
COVERAGE = {
    "lookup": "Answer what is asked; no background methodology or checklist section.",
    "other": "Cover every checklist item with its most important figures, dates, names, ratios, thresholds and conditions (with units and the period they refer to).",
}


class Trace:
    def __init__(self):
        self.steps: list[dict] = []
        self.timings: dict[str, float] = {}

    @contextmanager
    def step(self, name: str, description: str, inp):
        lf = llm.langfuse()
        ctx = lf.start_as_current_observation(name=name, as_type="span", input=inp) if lf else nullcontext()
        entry = {"step_name": name, "description": description, "input": inp, "output": None}
        t = time.time()
        with ctx as span:
            yield entry
            if span is not None:
                span.update(output=entry["output"])
        self.timings[name] = round(time.time() - t, 2)
        self.steps.append(entry)


def _catalogue(col: store.Collection) -> str:
    lines = []
    for c in col.docs.values():
        sib = "; ".join(f"{s['doc_id']}: {s['difference']}" for s in c.get("siblings", []) if s.get("difference"))
        lines.append(json.dumps({
            "doc_id": c["doc_id"], "file": c["filename"], "title": c.get("title"), "type": c.get("doc_type"),
            "entity": c.get("entity"), "period": c.get("period"), "date": c.get("date"), "status": c.get("status"),
            "jurisdiction": c.get("jurisdiction"), "pages": c.get("n_pages"), "summary": c.get("summary"),
            "topics": (c.get("topics") or [])[:12], **({"siblings": sib} if sib else {}),
        }, ensure_ascii=False))
    return "\n".join(lines)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " …"


def write_answer(col: store.Collection, question: str, qtype: str, facets: list[str], missing: list[str],
                 evidence_units: list[str], usage: llm.Usage, *, words: int | None = None, model: str | None = None,
                 extra_rules: str = "") -> str:
    """The answer step on its own (also used by scripts/replay_answer.py to A/B answer settings
    on fixed evidence)."""
    ev, used = [], 0
    for n, uid in enumerate(evidence_units):
        u = col.units[col.unit_idx[uid]]
        block = f"[E{n+1}] {col.docs[u['doc_id']]['filename']} p{u['page']} — {u['header']}\n{u['text']}"
        if used + len(block) > MAX_EVIDENCE_CHARS:
            break
        used += len(block)
        ev.append(block)
    missing_txt = (("Documents the question refers to that are NOT in the collection: " + "; ".join(missing)
                    + "\nSay so explicitly and answer only from what is available.\n\n") if missing else "")
    style = STYLE[qtype].format(words=words or LIMITS[qtype]["words"]) + (("\n" + extra_rules) if extra_rules else "")
    return llm.chat(ANSWER_SYSTEM, ANSWER_PROMPT.format(
        question=question, facets="\n".join(f"{i+1}. {f}" for i, f in enumerate(facets)), missing=missing_txt,
        evidence="\n\n".join(ev), coverage=COVERAGE["lookup" if qtype == "lookup" else "other"], style=style),
        model=model or config.ANSWER_MODEL, json_mode=False, max_tokens=4000, usage=usage, name="answer")


def run_query(collection: str, question: str) -> dict:
    t0 = time.time()
    col = store.load(collection)
    usage = llm.Usage()
    tr = Trace()
    lf = llm.langfuse()
    root = lf.start_as_current_observation(name="kb_query", as_type="chain", input={"collection": collection, "query": question}) if lf else nullcontext()
    with root as root_span:
        result = _pipeline(col, question, usage, tr)
        result["usage"] = usage.to_dict()
        result["latency_seconds"] = round(time.time() - t0, 2)
        result["step_seconds"] = tr.timings
        result["trace"] = tr.steps
        if root_span is not None:
            root_span.update(output={"answer": result["answer"][:4000], "n_chunks": len(result["chunks"]), **result["usage"]})
            result["langfuse_trace_id"] = lf.get_current_trace_id()
    if lf:
        lf.flush()
    return result


def _pipeline(col: store.Collection, question: str, usage: llm.Usage, tr: Trace) -> dict:
    # 1. route --------------------------------------------------------------------------
    with tr.step("route",
                 "LLM reads the auto-built document catalogue (entity, type, period, status, look-alike notes) and picks the documents, question type, checklist and search queries",
                 {"question": question, "n_documents_in_catalogue": len(col.docs)}) as st:
        try:
            plan = llm.chat(ROUTE_SYSTEM, ROUTE_PROMPT.format(question=question, catalogue=_catalogue(col)),
                            max_tokens=5000, usage=usage, name="route")
        except RuntimeError as e:  # degrade to collection-wide search rather than fail
            plan = {"confidence": "low", "error": str(e)[:300]}
        doc_ids = [d["doc_id"] for d in plan.get("documents", []) if isinstance(d, dict) and d.get("doc_id") in col.docs]
        qtype = plan.get("question_type") if plan.get("question_type") in LIMITS else "analysis"
        facets = [question] if qtype == "lookup" else ([f for f in plan.get("facets", []) if isinstance(f, str)] or [question])
        queries = [q for q in plan.get("search_queries", []) if isinstance(q, str)][:6]
        missing = [str(m) for m in plan.get("missing", []) if m]
        lim = LIMITS[qtype]
        st["output"] = plan

    # 2. retrieve -----------------------------------------------------------------------
    fallback = not doc_ids or plan.get("confidence") == "low"
    probes = [question] + queries + [f for f in facets if f != question]
    with tr.step("retrieve",
                 "Dense + BM25 search restricted to the routed documents (one search per query / checklist item, merged by best score); "
                 "each hit expanded to its small leaf section; near-duplicates collapsed; round-robin budget across documents. "
                 "Also searches the whole collection when routing is empty or low-confidence",
                 {"probes": probes, "doc_scope": doc_ids or "ALL", "fallback_to_collection": fallback}) as st:
        per_probe_k = max(10, 2 * lim["hits"] // len(probes))
        merged: dict[int, float] = {}
        for scope in ([doc_ids] if doc_ids else []) + ([None] if fallback else []):
            for pq in probes:
                for i, sc in col.search([pq], scope, k=per_probe_k):
                    merged[i] = max(merged.get(i, 0.0), sc)
        hits = sorted(merged.items(), key=lambda x: -x[1])[: lim["hits"]]
        rank = {col.units[i]["id"]: r for r, (i, _) in enumerate(hits)}

        ordered, seen, sigs = [], set(), set()
        for i, _ in hits:
            u = col.units[i]
            leaf = col.sections[u["section_id"]]["unit_ids"]
            for uid in ([u["id"]] + (leaf if len(leaf) <= lim["leaf"] else [])):
                if uid in seen:
                    continue
                seen.add(uid)
                sig = re.sub(r"\W+", "", col.units[col.unit_idx[uid]]["text"].lower())[:300]
                if sig not in sigs:  # repeated boilerplate collapses to one unit
                    sigs.add(sig)
                    ordered.append(uid)
        per_doc: dict[str, list[str]] = {}
        for uid in ordered:
            per_doc.setdefault(col.units[col.unit_idx[uid]]["doc_id"], []).append(uid)
        queues = sorted(per_doc.values(), key=lambda q: min(rank.get(u, 10_000) for u in q))
        cand: list[str] = []
        while len(cand) < lim["candidates"] and any(queues):
            for q in queues:
                if q and len(cand) < lim["candidates"]:
                    cand.append(q.pop(0))
        docs_used = sorted({col.units[col.unit_idx[u]]["doc_id"] for u in cand})
        st["output"] = {"n_hits": len(hits), "n_candidates": len(cand),
                        "per_document": {d: sum(1 for u in cand if u.startswith(d + ":")) for d in docs_used}}

    # 3. grade --------------------------------------------------------------------------
    facet_txt = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facets))
    scope_txt = "; ".join(col.docs[d].get("short_title") or col.docs[d]["filename"] for d in (doc_ids or docs_used))
    with tr.step("grade",
                 "Fast LLM grades every candidate in parallel batches: 2 = directly answers, 1 = needed context, 0 = not needed "
                 "(topic-only, boilerplate, wrong entity / period / version)",
                 {"n_candidates": len(cand), "batch_size": GRADE_BATCH}) as st:
        batches = [cand[b : b + GRADE_BATCH] for b in range(0, len(cand), GRADE_BATCH)]
        jobs = [dict(system=GRADE_SYSTEM, user=GRADE_PROMPT.format(
            question=question, facets=facet_txt, scope=scope_txt,
            strictness=LOOKUP_STRICTNESS if qtype == "lookup" else "",
            passages="\n\n".join(f"<passage id=\"{u}\">\n{_clip(col.units[col.unit_idx[u]]['search_text'], GRADE_PASSAGE_CHARS)}\n</passage>" for u in ids)),
            max_tokens=2000, usage=usage, name="grade") for ids in batches]
        grade: dict[str, int] = {}
        failed = 0
        for ids, res in zip(batches, llm.chat_many(jobs, concurrency=GRADE_CONCURRENCY)):
            if not isinstance(res, dict):
                failed += 1
                continue
            for r in res.get("results", []):
                if isinstance(r, dict) and r.get("id") in ids and str(r.get("grade")) in ("1", "2"):
                    grade[r["id"]] = int(r["grade"])
        if batches and failed == len(batches):  # LLM unavailable: an error, not a confident "nothing found"
            raise RuntimeError("evidence grading failed for every batch; check the LLM API key / budget")
        g2 = [u for u in cand if grade.get(u) == 2]
        g1 = [u for u in cand if grade.get(u) == 1]
        st["output"] = {"n_grade2": len(g2), "n_grade1": len(g1), "failed_batches": failed}

    # 4. answer -------------------------------------------------------------------------
    doc_order = lambda us: sorted(us, key=lambda u: col.unit_idx[u])  # noqa: E731 - document order reads best
    evidence_units = doc_order(g2 if len(g2) >= MIN_GRADE2["answer"] else g2 + g1)
    returned = doc_order(g2 if len(g2) >= MIN_GRADE2["chunks"] else g2 + g1)
    with tr.step("answer",
                 "Answer written only from the graded evidence (grade 2, plus grade-1 context when grade-2 is thin), bounded length, "
                 "[file pN] citations; abstains with no chunks when nothing is directly relevant",
                 {"n_evidence_units": len(evidence_units), "question_type": qtype, "word_budget": lim["words"],
                  "missing_documents": missing}) as st:
        if not g2:
            answer = "No information was retrieved for this question: the indexed documents do not appear to contain it."
            if missing:
                answer += " Not in the collection: " + "; ".join(missing) + "."
            returned = []
        else:
            answer = write_answer(col, question, qtype, facets, missing, evidence_units, usage)
        st["output"] = {"answer_chars": len(answer), "n_returned_chunks": len(returned)}

    # API-contract shape: content + metadata.file_name (as in the corpus) + 1-based page_numbers;
    # extra metadata fields are allowed and ignored by the harness
    chunks = []
    for uid in returned:
        u = col.units[col.unit_idx[uid]]
        chunks.append({
            "content": u["text"],
            "metadata": {
                "file_name": col.docs[u["doc_id"]]["filename"],
                "page_numbers": [u["page"]],
                "section": u["header"].split(" | ")[1] if " | " in u["header"] else "",
                "chunk_id": uid, "kind": u["kind"], "grade": grade.get(uid),
            },
        })
    return {"answer": answer.strip(), "chunks": chunks}
