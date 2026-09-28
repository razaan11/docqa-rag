# DocQA RAG

Ask questions about your PDFs and get answers grounded in the documents, with page-level citations.

A retrieval-augmented generation (RAG) service built with **FastAPI**, **Google Gemini**, **ChromaDB** and **Docker**.

## How it works

```
PDF ──► extract text per page ──► split into overlapping chunks (800 chars, 150 overlap)
                                          │
                                          ▼
                              Gemini embeddings ──► ChromaDB (cosine similarity)

Question ──► embed ──► top-k similar chunks ──► Gemini answers using ONLY those chunks
                                                 and cites them as [1], [2] with file + page
```

- **Grounded answers:** the prompt restricts the model to retrieved context and tells it to say "I don't know" when the answer isn't there, which reduces hallucination.
- **Citations:** every answer returns its source chunks with file name, page number and similarity score.
- **Idempotent ingestion:** chunk IDs are content hashes, so re-uploading a file updates it instead of duplicating it.
- **Offline tests:** a fake embedding model stands in for Gemini, so the test suite runs without an API key.

## Run locally

```bash
pip install -r requirements.txt
$env:GEMINI_API_KEY="your-key-here"     # Windows PowerShell; macOS/Linux: export GEMINI_API_KEY=your-key-here
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs to try the API in your browser.

## Run with Docker

```bash
docker build -t docqa-rag .
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key-here -v docqa-data:/data docqa-rag
```

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ingest` | Upload a PDF; returns the number of chunks indexed |
| POST | `/ask` | `{"question": "...", "k": 4}` → answer plus cited sources |
| GET | `/health` | Service status and number of indexed chunks |

## Tests

```bash
python -m pytest -q
```

## Configuration

| Variable | Default |
|---|---|
| `GEMINI_API_KEY` | required |
| `CHAT_MODEL` | `gemini-2.5-flash` |
| `EMBED_MODEL` | `gemini-embedding-001` |
| `CHROMA_PATH` | `./chroma_db` |

## Limitations and next steps

- Scanned (image-only) PDFs aren't supported yet; OCR would be needed.
- Fixed-size chunking; semantic or heading-aware chunking could improve retrieval.
- No re-ranking step; a cross-encoder re-ranker would sharpen the top results.
- No automated answer-quality evaluation yet.
