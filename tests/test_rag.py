"""Offline tests: a fake LLM stands in for Gemini so the pipeline runs without an API key."""

import hashlib

import pytest
from pypdf import PdfWriter

from app.rag import RAGPipeline, chunk_text


class FakeLLM:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors."""

    def embed(self, texts):
        vecs = []
        for t in texts:
            v = [0.0] * 64
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            vecs.append(v)
        return vecs

    def generate(self, prompt):
        return "stub answer"


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, size=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]  # consecutive chunks overlap


def test_chunk_rejects_bad_params():
    with pytest.raises(ValueError):
        chunk_text("abc", size=50, overlap=50)


def test_retrieval_ranks_relevant_chunk_first(tmp_path):
    rag = RAGPipeline(llm=FakeLLM(), db_path=str(tmp_path / "db"))
    rag.collection.upsert(
        ids=["a", "b"],
        embeddings=FakeLLM().embed(["refund policy allows returns within 30 days", "office opens at nine am"]),
        documents=["refund policy allows returns within 30 days", "office opens at nine am"],
        metadatas=[{"source": "policy.pdf", "page": 1}, {"source": "info.pdf", "page": 2}],
    )
    hits = rag.retrieve("what is the refund policy for returns", k=2)
    assert hits[0]["source"] == "policy.pdf"
    assert rag.ask("refund policy?")["answer"] == "stub answer"


def test_empty_pdf_ingests_nothing(tmp_path):
    pdf = tmp_path / "blank.pdf"
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.write(str(pdf))
    rag = RAGPipeline(llm=FakeLLM(), db_path=str(tmp_path / "db"))
    assert rag.ingest(str(pdf), "blank.pdf") == 0
