"""HTTP API: POST /index (zip + collection_name) and POST /query."""
from __future__ import annotations

import shutil
import tempfile
import threading
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import index, query, store

app = FastAPI(title="Smart Knowledge Base")
_index_lock = threading.Lock()  # one ingestion at a time; parsing already uses every core


class QueryIn(BaseModel):
    collection_name: str
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
async def index_endpoint(file: UploadFile = File(...), collection_name: str = Form(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "file must be a .zip of documents")
    tmp = Path(tempfile.mkdtemp())
    zpath = tmp / "upload.zip"
    with zpath.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    def work():
        with _index_lock:
            return index.index_zip(collection_name, zpath)

    try:
        stats = await run_in_threadpool(work)
    except zipfile.BadZipFile:
        raise HTTPException(400, "file is not a valid zip archive")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:  # e.g. LLM unavailable during card generation; previous collection untouched
        raise HTTPException(502, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"status": "success", "collection_name": collection_name, "stats": stats}


@app.post("/query")
async def query_endpoint(body: QueryIn):
    try:
        store.load(body.collection_name)
    except FileNotFoundError as e:
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
