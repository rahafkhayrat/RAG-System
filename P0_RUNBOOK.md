# P0 local verification runbook

## 1. Install dependencies (PowerShell)

```powershell
uv sync --group dev
uv sync --extra embeddings --group dev
```

The second command installs the optional local BGE-M3 embedding dependency. It may download model files on first use.

## 2. Configure the LLM (optional for offline tests)

Copy `.env.example` to `.env` and set `LLM_API_KEY` locally. Do not commit `.env` or paste API keys into chat.

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=YOUR_API_KEY
LLM_MODEL=gpt-4o-mini
RAG_EMBED_MODEL=BAAI/bge-m3
RAG_TOP_K=5
```

## Run the local API

```powershell
uv run uvicorn rag.api.app:app --host 127.0.0.1 --port 5000 --reload
```

## 3. Run automated tests

```powershell
uv run pytest -q
```

## 4. Run the offline benchmark

```powershell
uv run rag-benchmark --embedder hashing --top-k 3
```

For the multilingual embedding model:

```powershell
uv run rag-benchmark --embedder bge-m3 --top-k 3
```

## 5. Test actual generation (may incur provider charges)

```powershell
uv run rag-benchmark --embedder bge-m3 --top-k 3 --use-llm
```

Confirm that provider credentials and model access are valid. This benchmark is a tiny synthetic smoke test, not evidence of production answer quality.

## Important known limitations before production

- The vector store is in-memory; restarting the app loses indexed documents.
- The app factory requires real authenticated `scope_resolver` and `document_scope_resolver` callbacks from the host application. Do not replace them with request-body tenant/course IDs.
- Retrieval currently has no calibrated relevance/entailment threshold. A semantically related but insufficient excerpt may still be passed to the LLM. The prompt asks the LLM to abstain, but that is not a deterministic guarantee. The false-premise benchmark case needs explicit review with a real LLM and broader evaluation data.
- The benchmark's `hit@k` and `recall@k` evaluate retrieval only for six answerable questions; they do not certify unsupported-question refusal.
- The API does not itself provide authentication, rate limiting, persistent storage, or production deployment configuration.
