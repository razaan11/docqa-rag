"""FastAPI service exposing document ingestion and question answering."""

import os
import tempfile
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag import RAGPipeline

app = FastAPI(title="DocQA RAG", description="Ask questions about your PDFs with retrieval-augmented generation.")


@lru_cache
def _build_pipeline() -> RAGPipeline:
    return RAGPipeline()


def pipeline() -> RAGPipeline:
    try:
        return _build_pipeline()
    except RuntimeError as e:  # e.g. missing GEMINI_API_KEY: surface the reason instead of a bare 500
        raise HTTPException(500, str(e))


class Question(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": pipeline().collection.count()}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        count = pipeline().ingest(path, source=file.filename)
    finally:
        os.remove(path)
    if count == 0:
        raise HTTPException(422, "No extractable text found (scanned PDFs are not supported)")
    return {"file": file.filename, "chunks_indexed": count}


@app.post("/ask")
def ask(q: Question):
    return pipeline().ask(q.question, q.k)
