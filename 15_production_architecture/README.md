# 15 — Production Architecture

> Status: **Complete**

## What This Section Covers

- React + FastAPI + LangGraph architecture
- PostgreSQL checkpointer for durable state
- Redis rate limiting
- pgvector for RAG in the same DB
- JWT auth
- Input/output guardrails and PII redaction
- Horizontal scaling and production failure modes

## Files

| File | Purpose |
|------|---------|
| `01_architecture_overview.py` | Full ASCII stack diagram; why each component; request lifecycle (streaming + HITL); scaling; failure modes |
| `02_fastapi_langgraph.py` | All 5 endpoints (chat, stream, approve, history, health); JWT dep; SSE event generator; HITL resume; design decisions |
| `03_postgres_checkpointer.py` | AsyncPostgresSaver setup; DB schema; lifespan pool; time-travel; HITL checkpoint lifecycle; SQLite alternative; pgvector |
| `04_redis_rate_limit.py` | Sliding-window rate limiter; daily cost cap; JWT auth with streaming token workaround; input + output guardrails; session cache |
| `05_guardrails.py` | Runnable demo: length check, injection blocklist, PII redaction, LLM safety classifier, full pipeline |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
# No server needed — these print patterns and explanations
python 15_production_architecture/01_architecture_overview.py
python 15_production_architecture/02_fastapi_langgraph.py
python 15_production_architecture/03_postgres_checkpointer.py
python 15_production_architecture/04_redis_rate_limit.py

# This one actually runs (requires OPENAI_API_KEY)
python 15_production_architecture/05_guardrails.py
```

## Architecture at a Glance

```
Browser (React / EventSource)
  ↓ HTTPS / SSE
NGINX (TLS, proxy_buffering off)
  ↓
FastAPI (async)
  JWT auth → Redis rate limit → input guardrail → LangGraph → output guardrail
  ↓                                  ↓
PostgreSQL                        OpenAI API + Tool APIs
  - AsyncPostgresSaver                 (Jira / AWS / K8s)
  - pgvector (RAG)
  ↓
LangSmith (tracing, evals — zero code change)
```

## Key Rules

- **Always async** — use `ainvoke()` / `astream_events()`, never sync in FastAPI handlers
- **Connection pool at startup** — open once in `lifespan`, share across all requests
- **PostgreSQL over MemorySaver** — ACID checkpoints survive restarts and scale horizontally
- **Redis for rate limiting** — shared across instances; atomic INCR avoids race conditions
- **Guardrails in order** — input checks first (save LLM cost), output checks last (prevent leaks)
