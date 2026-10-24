# Smart Knowledge Base for Agents

Retrieval + answering engine for financial document collections. Instead of
"embed → top-k → rerank": an LLM **routes** the question to documents using an
auto-built catalogue (cards + look-alike notes), search runs only **inside** those
documents (one probe per checklist item, hits expanded to their sections), a fast LLM
**grades** every candidate, and one grounded **answer** call writes from the
directly relevant passages only — or abstains. See [`WRITEUP.md`](WRITEUP.md) for the architecture and results, and [`APPROACH.md`](APPROACH.md) for the research behind it.

## Layout

| Path | Purpose |
|---|---|
| `kb/parse.py` | PDF → page-bounded structural units + section tree (outline / font headings, ruled tables → markdown, Tesseract OCR for scanned pages, crash-safe process pool) |
| `kb/index.py` | Ingestion: parse (cached) → LLM document cards → look-alike detection + difference notes → contextual unit headers → embeddings |
| `kb/store.py` | Collection persistence; dense + BM25 search scoped to documents; section expansion |
| `kb/query.py` | Query pipeline: route → retrieve → grade → answer, per-step trace (also Langfuse spans) |
| `kb/llm.py` | OpenCode client: Claude models via the Anthropic Messages API (official SDK, thinking disabled), others via the OpenAI-compatible endpoint; JSON mode, retries, token/cost accounting, Langfuse generations |
| `kb/config.py` | Settings from `.env` / environment (models, keys, prices, workers) |
| `kb/api.py` | FastAPI: `POST /index`, `POST /query`, `GET /health` |
| `scripts/` | `serve.sh` (run the API), `index_folder.py`, `run_benchmark.py`, `evaluate.py` (organisers' notebook code, unmodified; `--dev-judge`), `replay_answer.py`, `make_gold_run.py`, `check_parse_coverage.py`, `preparse.py` |
| `Dockerfile`, `requirements.txt` | Deployment (Python 3.12 + tesseract-ocr, pinned dependencies) |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r pyproject.toml
brew install tesseract          # or: apt-get install -y tesseract-ocr
cp .env.example .env            # LANGFUSE_* keys; OPENCODE_API_KEY optional (the organiser key is the code default)
```

## Deploy

### Railway (public HTTPS endpoint)
1. Railway **Hobby plan** or higher (Free / Trial allow only 0.5–1 GB RAM and 1–2 vCPU — too small for indexing).
2. New Project → *Deploy from GitHub repo* → this repo. `railway.json` makes Railway build the `Dockerfile` and health-check `/health`.
3. Service → *Volumes* → add a volume mounted at **`/data`** (collections and caches survive redeploys).
4. Service → *Variables*: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (optional: `OPENCODE_API_KEY`, `WORKERS`).
5. Service → *Settings → Networking* → *Generate Domain* → `https://<name>.up.railway.app` is the endpoint.

Railway closes requests that send no data for 5 minutes and caps them at 15 minutes, so `/index` streams a newline every 15 s while it works and ends with the JSON result; if indexing outlasts ~13 minutes the final JSON says `"status": "indexing"` and `GET /index/status?collection_name=…` reports completion.

### Docker / plain server
```bash
docker build -t smart-kb .
docker run -d --env-file .env -p 8000:8000 -v kb-data:/data smart-kb
# or, on a Linux server with Python 3.12 + tesseract-ocr installed:
./scripts/serve.sh
```

Expected ingestion time: ~15 min cold for ~7,000 pages on 8 cores (state ~20–25 min for headroom); re-indexing unchanged files is near-instant thanks to the parse / embedding caches. `/index` returns 400 at once for a bad or PDF-less zip and 409 while another indexing job runs; otherwise it streams and its final JSON is `{"status": "success" | "error" | "indexing", ...}` (an LLM outage while building cards ends in `"error"` and leaves any existing collection untouched). `/query` returns 404 for an unknown collection, 409 while it is still being indexed, and 502 if the LLM is unavailable.

## Run the API

```bash
.venv/bin/uvicorn kb.api:app --host 0.0.0.0 --port 8000
```

```bash
# index a zip of PDFs into a named collection
curl -F "file=@docs.zip" -F "collection_name=train" http://localhost:8000/index

# ask a question
curl -H "Content-Type: application/json" \
     -d '{"collection_name": "train", "query": "Who chairs the Risk Management Committee of Infosys?"}' \
     http://localhost:8000/query
```

`/query` returns `answer`, `chunks` (each with `document`, `page`, `section`, `text`),
`trace` (array of `{step_name, description, input, output}`) and `metadata`
(token usage, cost, latency, per-step seconds, Langfuse trace id).

## Dev loop

```bash
# the dev scripts expect the organisers' Train/ folder (not in this repo) at the project root
.venv/bin/python scripts/index_folder.py train Train/Documents
.venv/bin/python scripts/run_benchmark.py train runs/r1.jsonl --every 5
.venv/bin/python scripts/evaluate.py runs/r1.jsonl      # organisers' notebook code, unmodified
```
