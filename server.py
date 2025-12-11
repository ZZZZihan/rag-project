"""FastAPI backend for the RAG service."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from rag_core import DATA_DIR, rag_service


app = FastAPI(title="RAG Knowledge System", version="0.1.0")


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    DATA_DIR.mkdir(exist_ok=True)
    target_path = DATA_DIR / Path(file.filename).name

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_added = rag_service.ingest_document(str(target_path))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "filename": file.filename,
        "status": "success",
        "chunks_added": chunks_added,
    }


@app.post("/chat")
async def chat(request: QueryRequest) -> dict:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = rag_service.get_answer(request.question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
