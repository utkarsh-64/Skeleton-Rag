# Plug-and-play RAG

Drop-in Python module for PDF and Word (.docx) question answering with Gemini.
SQLite stores extracted text and embeddings locally. Requires Python 3.12+.

## Add to an existing Python project: three files

Copy `rag.py`, `requirements.txt`, and `.env.example` into your project. Merge the
requirements into your existing dependency file if you already have one. Copy
the example to `.env`, or merge its variables into your existing `.env` without
overwriting existing settings.

```sh
pip install -r requirements.txt
```

Set these variables (all other core settings have defaults):

```dotenv
GEMINI_API_KEY=your-key
PROJECT_NAME=my-project
RAG_PROMPT="You are an HR assistant. Explain company policies concisely."
```

Use it from your existing backend:

```python
from rag import RAG

rag = RAG()  # Reads .env; create once per application process.
rag.ingest("documents/handbook.pdf")  # Index on upload, not on every question.
result = rag.ask("How many days of leave do employees get?")
print(result["answer"])
print(result["sources"])
rag.close()  # At application shutdown.
```

For uploaded bytes, use `rag.ingest_bytes(contents, "handbook.docx")`. Use
`rag.list_documents()` and `rag.delete(document_id)` to manage the index.
`data/rag.sqlite3` is created automatically. Preserve this file across restarts.

`RAG(env_file="/path/to/project.env")` selects another env file. Environment
variables override file values; paths are relative to the process working
directory, so set an absolute `DATABASE_PATH` if your launch directory varies.

The calls are synchronous. In FastAPI, call them in normal `def` routes, or use
`await run_in_threadpool(rag.ask, question)` from an async route. Route your UI to
your existing backend and return the result as JSON. Keep Gemini credentials on
the backend. Validation raises `ValueError`, unusable model responses raise
`RAGError`, and Gemini SDK/network exceptions propagate to your backend.

Python source cannot be imported directly into Node, Java, or other runtimes.
Those backends would need a Python HTTP adapter, which is not included here.

Responses contain `answer` and `sources` with filename, document ID, chunk index,
excerpt, similarity score, citation number, and PDF page number. DOCX pages are
`null` because page numbers depend on rendering. Sources are retrieved evidence;
they do not independently verify the model's answer or citation accuracy.

## Project configuration

Set `PROJECT_NAME`, credentials, generation model, prompt path, chunk settings,
and retrieval settings in `.env`. `MIN_SCORE` is a cosine-similarity cutoff, not a
confidence probability; tune it using representative questions.

The embedding integration targets `gemini-embedding-001` with 768 dimensions and
retrieval task types. `EMBEDDING_MODEL` may only select models with that same API
contract; changing to another embedding family may require code changes.
Project name, embedding model, chunk size, and overlap isolate stored indexes.
Changing any of those requires uploading documents again. Old indexes stay on
disk. Exact duplicate uploads within an index reuse the first filename and avoid
embedding calls. Changed files create new documents: delete the old ID to replace.

## Scope and limits

- Supports text PDFs and `.docx` body paragraphs/tables. Convert legacy `.doc` first.
  Scanned PDFs/images need OCR; headers, footers, text boxes, and nested Word tables
  are not extracted. Mixed scanned/text PDFs may have missing image-only content.
- Ingestion is synchronous, batches 20 chunks per embedding request, and commits
  only after all embeddings succeed. Failed retries can consume additional quota.
- SQLite retrieval scans all stored vectors: intended for small collections and
  modest traffic. No job queue, chat history, streaming, or multi-user permissions.
- Gemini free-tier quotas/model availability depend on your account. Quota errors are
  surfaced to callers; there are no automatic retries. Document text is sent to
  Google for embeddings and answering. Check Google's free-tier data terms before
  uploading private material.

Official references: [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing),
[embeddings](https://ai.google.dev/gemini-api/docs/embeddings).

## Tests

After installing `requirements.txt`, run:

```powershell
python -m unittest discover -s tests -v
```

Tests use a fake Gemini client; no API key, network calls, or quota is required.
