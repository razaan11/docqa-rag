"""Core RAG pipeline: PDF loading, chunking, embedding, vector storage and grounded answering."""

import hashlib
import os
from dataclasses import dataclass

import chromadb
from pypdf import PdfReader

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
DB_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int


def load_pdf(path: str) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page that has extractable text."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping windows, preferring to break on whitespace."""
    if size <= overlap:
        raise ValueError("chunk size must be larger than overlap")
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def build_chunks(path: str, source: str) -> list[Chunk]:
    chunks = []
    for page, text in load_pdf(path):
        for n, piece in enumerate(chunk_text(text)):
            # Stable IDs make re-ingesting the same file an upsert instead of a duplicate.
            cid = hashlib.sha1(f"{source}:{page}:{n}:{piece[:50]}".encode()).hexdigest()
            chunks.append(Chunk(cid, piece, source, page))
    return chunks


class GeminiClient:
    def __init__(self, api_key: str | None = None):
        from google import genai

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY in your environment or .env file")
        self.client = genai.Client(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), 50):  # batch to stay within request limits
            res = self.client.models.embed_content(model=EMBED_MODEL, contents=texts[i:i + 50])
            vectors.extend(e.values for e in res.embeddings)
        return vectors

    def generate(self, prompt: str) -> str:
        return self.client.models.generate_content(model=CHAT_MODEL, contents=prompt).text


class RAGPipeline:
    def __init__(self, llm=None, db_path: str = DB_PATH):
        self.llm = llm or GeminiClient()
        self.db = chromadb.PersistentClient(path=db_path)
        self.collection = self.db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    def ingest(self, path: str, source: str) -> int:
        chunks = build_chunks(path, source)
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=self.llm.embed([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )
        return len(chunks)

    def retrieve(self, question: str, k: int = 4) -> list[dict]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(query_embeddings=self.llm.embed([question]), n_results=k)
        return [
            {"text": doc, "source": meta["source"], "page": meta["page"], "score": round(1 - dist, 3)}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def ask(self, question: str, k: int = 4) -> dict:
        hits = self.retrieve(question, k)
        if not hits:
            return {"answer": "No documents have been ingested yet.", "sources": []}
        context = "\n\n".join(f"[{i + 1}] ({h['source']}, p.{h['page']}) {h['text']}" for i, h in enumerate(hits))
        prompt = (
            "Answer the question using ONLY the context below. Cite sources as [1], [2], etc. "
            "If the context does not contain the answer, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return {"answer": self.llm.generate(prompt), "sources": hits}
