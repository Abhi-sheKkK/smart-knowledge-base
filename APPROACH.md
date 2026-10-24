# Smart Knowledge Base for Agents — Findings, Approach & Implementation Plan

_Status (2026-10-24): end-to-end pipeline implemented and running on the training set; tuning in progress. §1–3 are the original findings; §4–5 now describe what is **built** (differences from the original design are marked); §8 has implementation status, measurements and lessons._

---

## 1. Problem recap

Build `/index` (zip + `collection_name`) and `/query` (`collection_name` + `query` → answer, cited chunks, step trace) over an unorganised dump of financial PDFs. Scored on a **hidden** corpus and question set:

- Answer precision / recall (LLM judge, point-by-point vs gold answer)
- Context precision / recall (LLM judge of each returned chunk's relevance)
- Cost per query and latency (Langfuse)

Target bar is 0.95 on all four; plain hybrid / rerank / HyDE / CRAG / GraphRAG / ReAct loops are explicitly not acceptable as the core architecture.

---

## 2. Findings from the training data and evaluator

### 2.1 How the evaluator actually scores (`Train/Evaluation/Evaluation.ipynb`)

| Metric | Formula in code | Implication |
|---|---|---|
| Answer precision | `(ai_points − extra_points) / ai_points` | Every unsupported claim we add costs points. No padding, no speculation. |
| Answer recall | `matched_benchmark_points / benchmark_points` | Must cover every facet the gold answer covers. |
| Context precision | `judge_relevant / len(retrieved_chunks)` | Every returned chunk must be clearly relevant **to the query as read by the judge**. |
| Context recall | `judge_relevant / len(benchmark_chunks)` | Count-based, **not** matched against gold chunks. Gold count averages ~88/question, so a top-5 retriever is structurally capped very low. Not clipped at 1 — we deliberately do **not** exploit this (no artificial over-splitting); we match the benchmark's natural granularity. |

- Judge model: `deepseek-v4.1-flash` via OpenCode, temperature 0, JSON output; all chunks judged in **one** call (max_tokens 10000) → keep chunk count sane (≲150) and each chunk self-explanatory.
- **Found while running it:** the judge model is a *reasoning* model. On long answers (and on the ~5,000-char gold answers) it spends the whole 10,000-token budget on hidden reasoning and returns empty or truncated JSON — 18/20 of our first answers could not be scored. With reasoning disabled it scores, but becomes unreliable on very long point lists (>~150 points it sometimes flags *every* answer point as unsupported while matching most gold points). Practical consequence: answers should be close to gold length. See §8.4.
- Questions with 0 gold chunks (e.g. B32) give context recall 0 regardless; answer must abstain cleanly.

### 2.2 Benchmark shape (`Train/Benchmarks/benchmark_100.jsonl`)

| Property | Value |
|---|---|
| Questions | 100 — families A (36, company credit analysis), B (32, multi-doc regulation), C (32, quant / accounting methods) |
| Docs per question | mean 3.3, max 9 (A ≈ 1.9, B ≈ 4.6, C ≈ 3.6) |
| Gold chunks per question | mean 88, range 0–352 |
| Gold chunk size | mean ~860 chars (5 – 14,447); always **one page**; structure-labelled (`Section_n`, `table_…`, `math_…`, heading text) |
| Gold answer length | mean ~5,000 chars; markdown with bottom line, sections, tables, `[doc p.X]` citations |
| Abstention | ~16% of gold answers explicitly state what the evidence does **not** contain (A33 image-only table, A36 asks for a 2025 report that isn't in the corpus, B32 "No information was retrieved") |
| Short factoid questions | A33–A35, B30, B32, C31 — 1–3 sentence answers |

### 2.3 Corpus (`Train/Documents/`, 35 PDFs, 4 – 1,982 pages)

- **Deliberate near-duplicates / distractors**: `BaselFramework.pdf` (1,982 pp) vs `BCBS-Basel-Endgame.pdf`; `91fr14952.pdf` vs `91fr14908.pdf`; Metal Coatings FY2025 vs FY2026; `fact-sheet-SUMMARY.pdf` vs the referenced `fact-sheet-20260319.pdf`.
- **Unused by any training question** (after resolving the renames below): `BaselFramework.pdf`, `91fr14952.pdf` → good material for our own held-out synthetic questions.
- **Benchmark file names that didn't match the folder — resolved:**
  - `fact-sheet-20260319.pdf` = `fact-sheet-SUMMARY.pdf` (renamed; 83/83 gold chunks match verbatim).
  - `Guidelines on Management of Problem Loans.pdf` = `SAMA_EN_3575_VER1.pdf` (SAMA, Jan 2020). It is a **scanned image PDF with no text layer** → ingestion **must OCR** image-only pages.
  - `2026-05960.pdf` was genuinely missing; downloaded from GovInfo (Federal Register 91 FR 15332, 27 Mar 2026, OCC/Fed/FDIC standardized-approach proposal) to `Train/Downloaded_Missing/2026-05960.pdf`. 121 pages; 385/445 gold text passages found verbatim, 379 on the same page number.
- **Implication:** file names are not reliable identifiers (renamed files, Federal Register numbers, scanner names). Document identity must come from content (the cards), and the ingester must handle scanned PDFs.
- Mixed content: narrative text, financial tables, formulas (Heston, Markov-functional, RiskMetrics), regulatory paragraphs, some image-only tables.

---

## 3. Research findings

| Theme | Evidence | Source |
|---|---|---|
| Picking the right document dominates | FinanceBench: shared vector store 19% vs per-document store 50% vs long context 79% | [FinanceBench, arXiv 2311.11944](https://ar5iv.labs.arxiv.org/html/2311.11944) |
| LLM document routing on (entity, year) | HDRR: failure rate 22.5% → 6.4%, correctness 49.0% → 67.7% vs chunk RAG; falls back to full-corpus search when routing fails | [arXiv 2603.26815](https://arxiv.org/html/2603.26815v3) |
| Same idea, practitioner | "Dumber PageIndex": LLM extracts company + years, retrieve within doc → 82% FinanceBench | [r/Rag](https://www.reddit.com/r/Rag/comments/1skst0b/i_tried_building_a_dumber_version_of_pageindex/) |
| Metadata disambiguates near-identical boilerplate across companies/years | Company / fiscal year / section metadata; pre-filter beats post-filter | [arXiv 2601.11863](https://arxiv.org/html/2601.11863v1), [arXiv 2510.24402](https://arxiv.org/html/2510.24402v1) |
| Near-duplicate handling | MinHash at ingest to group same-skeleton docs; collapse versions before generation ("model quietly blends"); "deprecated wins" failure mode | r/Rag [boilerplate](https://www.reddit.com/r/Rag/comments/1wvc700/boilerplate_document_retrieval/), [6 failure types](https://www.reddit.com/r/Rag/comments/1ve5ur8/the_correct_document_ranked_15th_out_of_85_a/), [r/LangChain](https://www.reddit.com/r/LangChain/comments/1toywbx/what_do_you_check_before_retrieved_docs_enter/) |
| Structure-based retrieval beats top-k for completeness | LAD-RAG: layout/section graph built at ingest + "expand to section community" → >90% perfect evidence recall with no top-k tuning; baselines need k = 22–27 to match | [arXiv 2510.07233](https://arxiv.org/html/2510.07233v2) |
| Locate-then-read | DeepRead: search only locates, then read the section in order via ToC → 82.7% FinanceBench, +10 pts avg over agentic search | [arXiv 2602.05014](https://arxiv.org/html/2602.05014v1) |
| Tree / summary indexes | RAPTOR ≈ +20 pts on QuALITY; PageIndex claims 98.7% FinanceBench (vendor-reported, unaudited) | [RAPTOR](https://arxiv.org/pdf/2401.18059), [PageIndex](https://pageindex.ai/), [critique](https://sjramblings.io/pageindex-deep-dive-vectorless-rag/) |
| PageIndex cost warning | Practitioners report 30–120 s per query from per-node LLM calls | [r/Rag](https://www.reddit.com/r/Rag/comments/1rds60k/anyone_tried_pageindex/) |
| Structure-aware chunking | Paragraph-group chunking nDCG@5 ≈ 0.459 vs fixed-size < 0.244 | [arXiv 2603.06976](https://arxiv.org/html/2603.06976v1) |
| Contextual headers on chunks | −35% to −49% retrieval failures | [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) |
| Adding more chunks hurts | "Chunk expansion" raised hallucination 14.7% → 22.2% | [arXiv 2510.24402](https://arxiv.org/html/2510.24402v1) |
| Filter before answering | >60% of top-100 candidates irrelevant in TREC; filter-then-rank | [AFR-Rank](https://www.sciencedirect.com/science/article/abs/pii/S0306457325001736) |
| Table extraction | Docling TEDS 0.887 vs PyMuPDF4LLM 0.401 (PyMuPDF ~8× faster) | [pdfmux benchmark](https://pdfmux.com/blog/pdfmux-vs-pymupdf-vs-marker-vs-docling/) |
| Claim verification | CoVe reduces hallucination; over-strict verification causes evasive answers (recall loss) | [CoVe](https://arxiv.org/abs/2309.11495), [arXiv 2507.20136](https://arxiv.org/pdf/2507.20136) |
| RAG vs long context routing | Self-Route: answer from RAG, escalate to long context only if unanswerable | [Li et al. 2407.16833](https://arxiv.org/pdf/2407.16833) |

_Research caveat: Reddit content is from search-result excerpts only (Firecrawl cannot scrape Reddit)._

---

## 4. Approach

**Core idea:** retrieval is a *navigation* problem, not a similarity problem. Resolve **which document** (cards), then **which sections** (stored ToC tree), then **every relevant piece in them** (structural expansion), then let a cheap model **filter and extract facts** before a single answer call. Vector/BM25 search is used only *inside* the routed documents as a locator.

### 4.1 Indexing (`POST /index`) — implemented in `kb/parse.py`, `kb/index.py`

Legend: ✅ built as designed · 🔁 built differently (why) · ⏳ not built yet

1. 🔁 **Unzip → per-file parse with PyMuPDF** (pinned **1.26.5**), Docling dropped for speed. Page extraction runs in a crash-safe process pool (`PagePool`): MuPDF can segfault on odd pages, so a dead worker rebuilds the pool and the affected pages are retried one by one, then without table detection; a page that still fails is kept empty instead of aborting ingestion. ✅ **Pages with no text layer are OCR'd** with Tesseract (220 dpi).
2. ✅ **Structure-aware units**: split along headings; ruled tables (`find_tables`, only on pages with vector lines) kept whole as markdown; running headers/footers removed (lines repeated on ≥25% of pages); never cross a page; ~250–2,200 chars (target 1,200). Text is NFKC-normalised (ligatures `ﬁ`→`fi`) and split thousands separators re-joined (`60, 000`→`60,000`).
3. 🔁 **Section tree** per document from the PDF outline when it has ≥5 entries, otherwise font-size / bold heading detection (dominant span size, wrapped headings merged); OCR pages use a numbering / ALL-CAPS heading detector. **No LLM section summaries** — headings + page ranges + search-hit counts proved enough for section selection and keep ingestion cheap.
4. ✅ **Document card** per document (one LLM call on opening pages + outline + 2 sampled pages): title, short_title, entity, issuer, doc_type, period, date, jurisdiction, status, topics, summary.
5. 🔁 **Look-alike detection**: MinHash (word 5-gram shingles, LSH threshold 0.15) **or** same normalised entity + doc type → an LLM writes a "how to tell these apart" note stored on both cards. (No embedding-based similarity; not needed.)
6. ✅ **Contextual header** on every unit: `[short_title | section path (last 3) | pN]`, used for embedding, BM25 and in returned chunks.
7. 🔁 **Per-collection stores as files** (`data/collections/<name>/`: `cards.json`, `sections.json`, `units.jsonl`, `vectors.npy`, `meta.json`), built in a temp dir and swapped in atomically; BM25 (`rank_bm25`) built in memory at load. No SQLite/FAISS — 24k units fit comfortably in numpy.
8. ⏳ Key-figures table (stretch) — not built.
9. ➕ **Caches**: parse cache keyed by file SHA-1 + parser version + PyMuPDF version; embedding cache keyed by (model, text hash). Re-indexing unchanged files skips parsing and embedding.

### 4.2 Query (`POST /query`) — implemented in `kb/query.py`

| # | Step (`step_name` in trace) | Status | What it does | Model / cost |
|---|---|---|---|---|
| 1 | `query_planning_and_document_routing` | 🔁 (analysis + routing merged into one call to save latency) | LLM reads the question + the whole card catalogue (with look-alike notes) → question type (lookup / analysis / comparison), facet checklist, documents (+ why), rejected look-alikes (+ why), **missing documents**, 3–6 search queries, confidence. For lookups the checklist is the question itself. | 1 call |
| 2 | `locator_search` | ✅ | Dense (MiniLM) + BM25, reciprocal-rank fusion, restricted to routed documents. **Fails open**: if routing is empty or low-confidence, also searches the whole collection and adds the top-3 documents by hit mass. | no LLM |
| 3 | `section_selection` | ✅ | LLM reads a bounded outline (≤140 lines/doc: shallow levels + every section with hits and its ancestors; each line shows pages, unit count, hit count) and picks sections; told to cover every checklist item and that broad review questions need many sections. | 1 call |
| 4 | `structural_expansion` | ✅ | Union of: every unit in chosen sections (very large sections → best-ranked 80), top-60 hits, each hit's small leaf section (≤6 units). Near-identical units collapsed. **Per-document round-robin budget** (lookup 60 / analysis 260 / comparison 320) so one huge routed document can't crowd out the others. | no LLM |
| 5 | `evidence_filter` | 🔁 (relevance + facet tags only; no fact extraction — cut output tokens) | Fast model screens candidates in parallel batches of 10: relevant for *this* question (right entity / period / version)? which checklist items? Lookups get a stricter rule (only passages stating the fact). Only kept units are used and returned. | ⌈N/10⌉ parallel calls |
| 6 | `gap_fill` | ⏳ | (planned) one targeted search for checklist items with zero evidence — `facets_uncovered` is already computed and traced. | — |
| 7 | `llm_answer_generation` | ✅ | Answer only from kept units (≤120k chars, document order), `[file pN]` citations, explicit "documents do not state…" for gaps, missing documents stated. **Length bounded to gold-like size**: lookup = direct answer + related specifics from the same passage (40–150 words); analysis/comparison = bottom line + per-facet sections, ~350–700 words, ≤~30 facts. | 1 call, max 2,500 tokens |
| 8 | `claim_verification` | ⏳ | (planned) drop sentences whose figures don't appear in the cited unit. | — |

Returned `chunks` = units kept by step 5, each `{chunk_id, document, page, section, kind, text}` where `text` = contextual header + unit text. Response also carries `trace` (array of `{step_name, description, input, output}`) and `metadata` (tokens, cost, latency, per-step seconds, Langfuse trace id). Every step is a Langfuse span; every LLM call a Langfuse generation with usage and cost; parallel filter calls stay nested under the query trace.

**Reasoning control.** The model is a reasoning model: hidden reasoning costs tokens and latency, and on 7/36 card calls it exhausted `max_tokens` before writing any content. `reasoning_effort: "none"` fixed that until the provider started rejecting it; now no effort value is sent by default (`REASONING_EFFORT` in `.env` can set one), a rejected value is dropped automatically for the rest of the process, every call has generous `max_tokens` headroom (4–8k), and truncated JSON replies are retried with double the budget. Billed tokens of failed attempts are still counted.

### 4.3 How this handles the two failure modes

- **Near-duplicates**: decided at the *document* level using explicit attributes (year, version, status) and sibling-difference notes, before similarity search ever runs.
- **Chunk-size dilemma**: completeness comes from reading whole sections chosen from the document's own structure, not from a larger k; precision comes from the per-unit filter.

### 4.4 Generalisation safeguards

- No document names, categories, or folder structure hard-coded; cards, trees and sibling notes are derived per collection.
- Routing fails open to whole-corpus search.
- Validate on held-out families and on synthetic questions over the 4 unused documents (see §6).

---

## 5. Tech stack (as built)

| Concern | Choice |
|---|---|
| API | FastAPI + uvicorn (`kb/api.py`): `POST /index` (multipart zip + `collection_name`), `POST /query`, `GET /health`; blocking work in a thread pool; one ingestion at a time |
| Parsing | PyMuPDF **1.26.5** (2.0 segfaults in `find_tables` and drops inter-word spaces) + Tesseract OCR via `pytesseract` |
| Storage | Files per collection (JSON / JSONL / `.npy`), loaded into memory; `rank_bm25` built at load |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `fastembed` (ONNX, local, no per-query API cost). Quantised `bge-small` was ~8× slower on CPU (11 vs 98 texts/s) |
| LLM | OpenCode `deepseek-v4.1-flash` for every step (`FAST_MODEL` / `ANSWER_MODEL` configurable), reasoning disabled |
| Near-dup | `datasketch` MinHash + LSH |
| Observability | Langfuse SDK v5 (`start_as_current_observation`), spans per step, generations per LLM call |
| Eval | `scripts/evaluate.py` executes the notebook cells verbatim; optional `--dev-judge` (see §8.4) |
| Env | Python 3.12 via `uv`; config in `.env` (template `.env.example`) |

---

## 6. Implementation plan (12 h, 2 people)

**P1 = ingestion & retrieval owner, P2 = query / answer / eval owner.**

| Hours | P1 | P2 | Exit check |
|---|---|---|---|
| 0–1 | Repo skeleton, FastAPI `/index` `/query` stubs, collection layout | OpenCode client, Langfuse wiring, eval harness that wraps the notebook classes unmodified | Stub endpoints return contract-shaped JSON; one eval run on dummy output |
| 1–3 | Docling/PyMuPDF parsing → page-bounded structural units; section tree | Baseline query path (vector top-k → answer) to get a **reference score** on 20 questions | Units for all 35 PDFs; baseline numbers logged |
| 3–5 | Document cards, sibling detection + difference notes, contextual headers, per-doc vector + BM25 | `query_analysis` + `document_routing` (+ fallback), routing accuracy check against gold doc lists | Routing doc-level recall/precision measured on 100 Qs |
| 5–7 | Section summaries, `section_selection`, `locator_search`, `structural_expansion` | `evidence_filter` (batched, parallel), fact extraction schema | Candidate pool recall vs gold docs/pages; filter precision |
| 7–9 | Indexing speed / robustness (BaselFramework 1,982 pp), ingestion-time estimate | `answer_generation` per facet, abstention handling, `claim_verification`; trace array | Full pipeline on 30 Qs with all four metrics |
| 9–10.5 | Deploy to provided server; multi-collection test; cost & latency tuning (batch sizes, model choice) | Error analysis by family (A/B/C) and by stage; fix top failure classes **generically** | Held-out family scores within a few pts of tuned family |
| 10.5–12 | Langfuse export, final run on full dev set | 2–3 page write-up (architecture, decisions, cost/latency table) | Submission checklist complete |

### Stage-level metrics to log (beyond the four official ones)

- Document routing recall / precision vs gold doc list
- Page recall of candidate pool vs gold pages (before filter)
- Filter keep-rate and judge-agreement on a sample
- Tokens, cost, latency per step

### Overfitting guards

- Tune on families A + C, check B (and rotate).
- Write ~15 synthetic questions on `BaselFramework.pdf` and `91fr14952.pdf` (unused by training questions), including year-trap and missing-document cases.
- Any prompt change must be justified by a failure class, not a specific question.

---

## 7. Risks & open questions

| Risk / question | Mitigation |
|---|---|
| Scanned / image-only PDFs (SAMA guideline is fully scanned) | Detect pages with no text layer at ingestion and OCR them (e.g. Tesseract / `ocrmypdf`, or Docling's OCR); budget the extra ingestion time |
| `2026-05960.pdf` is an external download, not organiser-provided | Use it for dev evaluation only; mention it in the write-up |
| Context window / pricing of the OpenCode models unknown | Check on day; design keeps per-call context small (cards, ToC, 20-unit batches) |
| Ingestion time on very large docs (1,982 pp) | PyMuPDF fast path, parallel per doc, section summaries only above a size threshold; state a realistic ingestion estimate |
| Image-only tables (e.g. A33) | Detect empty-text regions; answer states the gap instead of guessing (optional OCR if time) |
| Judge variance | Run eval twice on key comparisons; compare deltas, not single runs |
| Over-strict verification lowering recall | Only drop claims whose figures/entities are absent from cited text; never drop whole sections |
| 0.95 on all four metrics may be unreachable | Optimise the ranking criterion (answer P/R) first; document cost/latency and architecture clearly for the fallback judging |
| **Official evaluator fails on long answers** (judge reasoning exhausts max_tokens) | Raise with organisers; keep answers near gold length; self-evaluate with `--dev-judge`, clearly labelled |
| Judge noise on long point lists | Bounded answer length; compare runs on the same 20 questions; look at deltas, not single scores |
| Very large look-alike documents dominating candidates (e.g. 1,982-page consolidated framework) | Per-document round-robin candidate budget (implemented); routing told to prefer the specific document/version the question names |

---

## 8. Implementation status (2026-10-24)

### 8.1 What exists

| Component | File(s) | State |
|---|---|---|
| PDF parsing, OCR, structural units, section tree | `kb/parse.py` | ✅ working on all 36 training PDFs |
| Ingestion: cards, look-alike notes, headers, embeddings, caches | `kb/index.py` | ✅ |
| Collection store + scoped hybrid search + section expansion | `kb/store.py` | ✅ |
| Query pipeline with trace + Langfuse | `kb/query.py` | ✅ steps 1–5, 7; ⏳ gap fill, claim verification |
| OpenCode client (JSON mode, retries, usage/cost, adaptive reasoning param) | `kb/llm.py` | ✅ |
| HTTP API `/index`, `/query`, `/health` | `kb/api.py` | ✅ written; ⏳ not yet exercised end-to-end via HTTP / not deployed |
| Dev tools | `scripts/index_folder.py`, `run_benchmark.py`, `evaluate.py`, `check_parse_coverage.py`, `preparse.py` | ✅ |
| Docs | `README.md`, this file | ✅ |
| Write-up, Langfuse export, deployment | — | ⏳ |

### 8.2 Measurements (training corpus: 36 PDFs, 7,091 pages)

| What | Value |
|---|---|
| Cold parse (8 cores, incl. OCR of 92 scanned pages) | **9.0 min** (541 s) |
| Cards + look-alike notes + embeddings (24,013 units) | **5.6 min**; 48 LLM calls, ≈ $0.05 |
| **Total cold ingestion** | **≈ 15 min** (state ~20 min to organisers for headroom); re-index of unchanged files: seconds (caches) |
| Parse quality: gold-chunk tokens found on the same page | **0.943** over 7,797 gold chunks |
| Query latency (20-question sample) | 36 s → **23 s** after bounding answer length |
| Query cost | ≈ $0.02–0.04 at the conservative $0.30 / $1.20 per M tokens (published prices for this model range down to ~$0.05 / $0.21) |

### 8.3 Score history (same 20 questions: every 5th benchmark question, all three families)

| Run | Change | Ans P | Ans R | Ctx P | Ctx R (capped at 1) | Judge |
|---|---|---|---|---|---|---|
| r1 | first full pipeline | 0.17 | 0.75 | 0.62 | 0.50 | dev, reasoning off |
| r2 | answers bounded to gold-like length | 0.28 | 0.71 | 0.64 | 0.45 | dev, reasoning off |
| r2 (rescored) | — | _pending_ | | | | dev, reasoning on, 32k budget |
| r3 | per-document candidate budget, broader section selection, tighter length | _pending_ | | | | dev, reasoning on, 32k budget |

r1/r2 scores used a judge with reasoning disabled, which proved erratic on long point lists; they are **not comparable** with later rows.

### 8.4 Lessons / surprises (worth keeping for the write-up)

1. **PyMuPDF 2.0 is unsafe**: `find_tables` segfaults on some pages (6 in the OCC handbook) and its `dict` extraction drops inter-word spaces ("Inthisdocument,wegive…"). A dead `multiprocessing.Pool` worker hangs `pool.map` forever (lost ~1 h). → pinned 1.26.5, crash-safe `PagePool`, parser/library version in the cache key.
2. **Embedding throughput matters for ingestion time**: quantised `bge-small` (fastembed) ran at 11 texts/s on this CPU (≈ 33 min for 24k units); MiniLM runs at 98 texts/s.
3. **The LLM is a reasoning model and the provider changes its knobs**: hidden reasoning exhausted `max_tokens` (7/36 cards came back empty); `reasoning_effort: "none"` fixed it — and was then **rejected by the provider mid-run**, zeroing a benchmark run. → reasoning param now optional and auto-dropped on rejection; generous `max_tokens` everywhere; truncated JSON retried with a doubled budget.
4. **The official evaluator cannot score long answers as shipped**: its judge (same reasoning model) uses up the notebook's 10,000-token budget decomposing ~5k-char answers (ours *and* the gold). For self-evaluation only, `scripts/evaluate.py --dev-judge` runs the notebook code unmodified but raises the judge request's `max_tokens` to 32,000 and timeout to 300 s at the HTTP layer; results are labelled. **To raise with organisers.**
5. **Answer length dominates answer precision**: first answers were ~4× gold length (median 19.7k vs 4.75k chars, 100–200 points vs ~45 gold points); most "extra" points were true facts the gold simply didn't select. Bounding length lifted precision 0.17 → 0.28 and cut latency 36 → 23 s. Lookup answers must still carry the closely related specifics (gold B30 lists the whole phase-in schedule, not just "65%").
6. **Big look-alike documents crowd the candidate pool**: the 1,982-page consolidated Basel framework and a 378-page sibling proposal filled the 220-unit cap on family-B questions → per-document round-robin budget.
7. **Context recall is capped at the source for broad questions**: A01 has 111 gold pages but only 64 candidate units reached the filter → section selection now told broad reviews need many sections; larger budgets for analysis/comparison.

### 8.5 Next steps

1. Score r3; compare with rescored r2.
2. Gap fill for uncovered checklist items; number-level claim check.
3. Full 100-question run; family hold-out check (tune on A+C, check B); synthetic questions on unused documents.
4. Exercise `/index` and `/query` over HTTP with a zip; deploy; Langfuse export; write-up.
