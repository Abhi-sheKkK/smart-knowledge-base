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
| `scripts/` | `tunnel.sh` (API + ngrok), `serve.sh` (run the API), `page_metrics.py` (contract context scoring), `index_folder.py`, `run_benchmark.py`, `evaluate.py` (organisers' notebook code, unmodified; `--dev-judge`), `replay_answer.py`, `make_gold_run.py`, `check_parse_coverage.py`, `preparse.py` |
| `Dockerfile`, `requirements.txt` | Deployment (Python 3.12 + tesseract-ocr, pinned dependencies) |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r pyproject.toml
brew install tesseract          # or: apt-get install -y tesseract-ocr
cp .env.example .env            # LANGFUSE_* keys; OPENCODE_API_KEY optional (the organiser key is the code default)
```

## Deploy

### ngrok from a laptop (the organisers' contract)
```bash
brew install ngrok
ngrok config add-authtoken <token>                 # free account, dashboard.ngrok.com
# claim your free static dev domain on the ngrok dashboard, then:
NGROK_DOMAIN=<your-domain>.ngrok-free.dev ./scripts/tunnel.sh   # starts API + tunnel, keeps the Mac awake
./scripts/tunnel.sh stop
```
Submit `https://<your-domain>.ngrok-free.dev` (no route). With a static domain the URL survives restarts; keep the machine awake and both processes running during evaluation.

### Docker / Railway / plain server
```bash
docker build -t smart-kb .
docker run -d --env-file .env -p 8000:8000 -v kb-data:/data smart-kb   # listens on $PORT if set
./scripts/serve.sh                                                      # Linux, Python 3.12 + tesseract-ocr
```
Railway: Hobby plan or higher, deploy this repo (`railway.json`: Dockerfile build, `/health` check), add a volume at `/data`, set the `LANGFUSE_*` variables, generate a domain.

### Behaviour that matters for the harness
- `/index` is synchronous: the first 2xx means the collection is ready. A bad or PDF-less zip returns 400; an LLM outage while building cards returns 502 and leaves any existing collection untouched. Tunnels close connections idle for 5 minutes, so if indexing is still running after 4 minutes the response starts (200) and streams keep-alive whitespace until the final JSON `{"status": "success", "collection_name", "documents_indexed", ...}` (a failure after that point is reported as `"status": "error"` in the body).
- `/query` returns 404 `{"detail": "unknown collection"}`, 200 with an honest answer and empty chunks when the documents do not contain the answer, and 502 if the LLM is unavailable.
- Expected ingestion time: ~15 min cold for ~7,000 pages on 8 cores (state ~20–25 min for headroom); unchanged files re-index in seconds (parse / embedding caches).

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

`/query` returns `answer`, `chunks` (each `{"content", "metadata": {"file_name", "page_numbers", ...}}`,
file names as in the corpus, 1-based pages), `trace` (array of `{step_name, description, input, output}`)
and `metadata` (token usage, cost, latency, per-step seconds, Langfuse trace id).

## Dev loop

```bash
# the dev scripts expect the organisers' Train/ folder (not in this repo) at the project root
.venv/bin/python scripts/index_folder.py train Train/Documents
.venv/bin/python scripts/run_benchmark.py train runs/r1.jsonl --every 5
.venv/bin/python scripts/evaluate.py runs/r1.jsonl      # organisers' notebook code, unmodified
```
