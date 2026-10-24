"""HTTP API per the organisers' contract: POST /index (zip + collection_name) and POST /query.

/index is synchronous: the first 2xx means the collection is ready. Tunnels such as ngrok
close a connection after 5 minutes without data, so if indexing is still running after
EARLY_RESPONSE_SECONDS the response is started (200) and keep-alive whitespace is streamed
until the final JSON (leading whitespace is valid JSON). Indexing that finishes before that
gets a normal response, including a real non-2xx status on failure."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import index, query, store

EARLY_RESPONSE_SECONDS = float(os.getenv("EARLY_RESPONSE_SECONDS", "240"))  # under ngrok's 5-min idle limit
HEARTBEAT_SECONDS = 20

app = FastAPI(title="Smart Knowledge Base")
_index_lock = threading.Lock()  # one ingestion at a time; parsing already uses every core


class QueryIn(BaseModel):
    collection_name: str
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


def _result(name: str, job: dict) -> tuple[int, dict]:
    if job["state"] == "success":
        s = job["stats"]
        return 200, {"status": "success", "collection_name": name, "documents_indexed": s.get("documents"), "stats": s}
    return job.get("code", 500), {"status": "error", "collection_name": name, "detail": job["detail"]}


def _run_job(name: str, pdfs: list[Path], skipped: list[str], job: dict):
    with _index_lock:
        try:
            stats = index.build_collection(name, pdfs)
            stats["skipped_non_pdf"] = skipped
            job.update(state="success", stats=stats)
        except RuntimeError as e:  # LLM unavailable during card generation; previous collection untouched
            job.update(state="error", code=502, detail=str(e)[:1000])
        except Exception as e:
            job.update(state="error", code=500, detail=f"{type(e).__name__}: {e}"[:1000])


@app.post("/index")
async def index_endpoint(file: UploadFile = File(...), collection_name: str = Form(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "file must be a .zip of documents")
    tmp = Path(tempfile.mkdtemp())
    zpath = tmp / "upload.zip"
    with zpath.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        pdfs, skipped = await run_in_threadpool(index.extract_zip, collection_name, zpath)
    except zipfile.BadZipFile:
        raise HTTPException(400, "file is not a valid zip archive")
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    job = {"state": "running", "started": time.time()}
    threading.Thread(target=_run_job, args=(collection_name, pdfs, skipped, job), daemon=True).start()

    # finished early: an ordinary response with its real status code
    while job["state"] == "running" and time.time() - job["started"] < EARLY_RESPONSE_SECONDS:
        await asyncio.sleep(1)
    if job["state"] != "running":
        code, body = _result(collection_name, job)
        return JSONResponse(body, status_code=code)

    # still running: start the 200 response and keep the connection alive until done
    async def stream():
        last = time.time()
        while job["state"] == "running":
            await asyncio.sleep(1)
            if time.time() - last >= HEARTBEAT_SECONDS:
                last = time.time()
                yield b" \n"
        yield json.dumps(_result(collection_name, job)[1]).encode()

    return StreamingResponse(stream(), media_type="application/json")


@app.post("/query")
async def query_endpoint(body: QueryIn):
    try:
        store.load(body.collection_name)
    except FileNotFoundError:
        raise HTTPException(404, "unknown collection")
    try:
        res = await run_in_threadpool(query.run_query, body.collection_name, body.query)
    except RuntimeError as e:  # LLM provider failure (the harness retries 5xx)
        raise HTTPException(502, str(e))
    return {
        "answer": res["answer"],
        "chunks": res["chunks"],
        "trace": res["trace"],
        "metadata": {k: res.get(k) for k in ("usage", "latency_seconds", "step_seconds", "langfuse_trace_id")},
    }
