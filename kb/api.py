"""HTTP API: POST /index (zip + collection_name), GET /index/status, POST /query.

/index validates the upload synchronously (400 on a bad / PDF-less zip), then builds the
collection in a background thread while the response streams a newline every few seconds
(leading whitespace is valid JSON). Proxies that close idle or long requests (e.g. Railway:
5 min without data, 15 min total) therefore see a live connection, and the final line is the
JSON result. If indexing outlasts INDEX_WAIT_SECONDS, the response says it is still running
and GET /index/status reports completion."""
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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import index, query, store

INDEX_WAIT_SECONDS = float(os.getenv("INDEX_WAIT_SECONDS", "780"))  # stay under a 15-minute proxy cap
HEARTBEAT_SECONDS = 15

app = FastAPI(title="Smart Knowledge Base")
_index_lock = threading.Lock()  # one ingestion at a time; parsing already uses every core
_jobs: dict[str, dict] = {}  # collection_name -> {"state": indexing|success|error, ...}


class QueryIn(BaseModel):
    collection_name: str
    query: str


def _job_payload(name: str, job: dict) -> dict:
    elapsed = round((job.get("finished") or time.time()) - job["started"], 1)
    if job["state"] == "success":
        return {"status": "success", "collection_name": name, "stats": job["stats"]}
    if job["state"] == "error":
        return {"status": "error", "collection_name": name, "detail": job["detail"], "elapsed_seconds": elapsed}
    return {"status": "indexing", "collection_name": name, "elapsed_seconds": elapsed,
            "detail": f"indexing continues in the background; poll GET /index/status?collection_name={name}"}


def _run_job(name: str, pdfs: list[Path], skipped: list[str], job: dict):
    with _index_lock:
        try:
            stats = index.build_collection(name, pdfs)
            stats["skipped_non_pdf"] = skipped
            job.update(state="success", stats=stats)
        except Exception as e:  # e.g. LLM unavailable during card generation; previous collection untouched
            job.update(state="error", detail=f"{type(e).__name__}: {e}"[:1000])
        finally:
            job["finished"] = time.time()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
async def index_endpoint(file: UploadFile = File(...), collection_name: str = Form(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "file must be a .zip of documents")
    if any(j["state"] == "indexing" for j in _jobs.values()):
        raise HTTPException(409, "another indexing job is running; retry when GET /index/status reports it finished")
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

    job = {"state": "indexing", "started": time.time()}
    _jobs[collection_name] = job
    threading.Thread(target=_run_job, args=(collection_name, pdfs, skipped, job), daemon=True).start()

    async def stream():
        last_beat = time.time()
        while job["state"] == "indexing" and time.time() - job["started"] < INDEX_WAIT_SECONDS:
            await asyncio.sleep(1)
            if time.time() - last_beat >= HEARTBEAT_SECONDS:
                last_beat = time.time()
                yield b"\n"  # keep-alive; whitespace before the JSON body is still valid JSON
        yield json.dumps(_job_payload(collection_name, job)).encode()

    return StreamingResponse(stream(), media_type="application/json")


@app.get("/index/status")
def index_status(collection_name: str):
    job = _jobs.get(collection_name)
    if job:
        return _job_payload(collection_name, job)
    try:
        return {"status": "success", "collection_name": collection_name, "stats": store.load(collection_name).meta}
    except FileNotFoundError:
        raise HTTPException(404, f"collection '{collection_name}' is not indexed")


@app.post("/query")
async def query_endpoint(body: QueryIn):
    try:
        store.load(body.collection_name)
    except FileNotFoundError as e:
        if _jobs.get(body.collection_name, {}).get("state") == "indexing":
            raise HTTPException(409, f"collection '{body.collection_name}' is still being indexed")
        raise HTTPException(404, str(e))
    try:
        res = await run_in_threadpool(query.run_query, body.collection_name, body.query)
    except RuntimeError as e:  # LLM provider failure
        raise HTTPException(502, str(e))
    return {
        "answer": res["answer"],
        "chunks": res["chunks"],
        "trace": res["trace"],
        "metadata": {k: res.get(k) for k in ("usage", "latency_seconds", "step_seconds", "langfuse_trace_id")},
    }
