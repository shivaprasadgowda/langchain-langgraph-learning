"""
Concept: Full production architecture for a LangGraph-powered chat application.

This file is a guided tour of the production stack — no API calls, no server.
Run it to see annotated architecture diagrams with component explanations.

Run:
    python 15_production_architecture/01_architecture_overview.py
"""


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


# ═══════════════════════════════════════════════════════════════════════════════
section("1. Full stack overview")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                        BROWSER (React)                          │
  │  ChatInput → EventSource(/chat/stream) → streaming token render │
  │  Auth: Bearer JWT in Authorization header (for POST requests)   │
  │  Thread: thread_id stored in localStorage / URL param          │
  └──────────────────────────┬──────────────────────────────────────┘
                             │ HTTPS
  ┌──────────────────────────▼──────────────────────────────────────┐
  │                   NGINX / CDN (Cloudflare)                      │
  │  TLS termination, static asset CDN, DDoS protection            │
  │  SSE: proxy_buffering off  (critical for streaming)            │
  └──────────────────────────┬──────────────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────────────┐
  │              FastAPI (Python, async, uvicorn)                   │
  │  POST /chat       → invoke, return final state (non-streaming)  │
  │  GET  /chat/stream → SSE streaming via astream_events()        │
  │  POST /chat/approve → resume HITL interrupt                     │
  │  GET  /chat/history → return thread state from checkpointer    │
  │                                                                  │
  │  Middleware stack (request → response):                         │
  │    1. JWT auth verification                                     │
  │    2. Redis rate limiter  (N requests/min per user)            │
  │    3. Input guardrail     (content filter, max length)         │
  │    4. LangGraph invoke / astream_events                        │
  │    5. Output guardrail    (PII redaction, content filter)      │
  └──────┬───────────────────┬───────────────────┬─────────────────┘
         │                   │                   │
  ┌──────▼──────┐   ┌────────▼────────┐  ┌──────▼──────────────────┐
  │  LangGraph  │   │   PostgreSQL    │  │   OpenAI API            │
  │  (compiled  │   │ (checkpointer:  │  │   (gpt-4o-mini)         │
  │   StateGraph│   │  thread state,  │  │                         │
  │   + tools)  │   │  checkpoints)   │  │   External tool APIs:   │
  └──────┬──────┘   └─────────────────┘  │   Jira, AWS, K8s       │
         │                               └─────────────────────────┘
  ┌──────▼──────┐   ┌─────────────────┐  ┌─────────────────────────┐
  │    Redis    │   │   pgvector      │  │   LangSmith             │
  │ (rate limit │   │  (vector store  │  │  (tracing, evals,       │
  │  cache,     │   │   for RAG,      │  │   debugging,            │
  │  sessions)  │   │   same Postgres │  │   feedback)             │
  └─────────────┘   │   with extension│  └─────────────────────────┘
                    └─────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("2. Why each component exists")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  FastAPI (not Flask/Django):
    - Native async support — essential for non-blocking LLM calls
    - Built-in Pydantic validation for request/response schemas
    - StreamingResponse / SSE without extra libraries
    - Automatic OpenAPI docs at /docs

  PostgreSQL (not Redis / SQLite for checkpoints):
    - ACID transactions — checkpoint writes are atomic
    - Persistent across server restarts (Redis loses data on restart)
    - pgvector extension: one DB for both chat state AND vector search
    - pg_notify: can push checkpoint events to listeners
    - SQLite alternative works for single-instance, not for horizontal scale

  Redis (not in-memory / DB for rate limiting):
    - Sub-millisecond reads — rate limit check adds <1ms per request
    - Atomic INCR + EXPIRE — race-condition-free sliding window
    - Shared across multiple FastAPI instances (unlike in-memory dict)
    - Also used for: session cache, SSE connection registry, job queue

  pgvector (not Pinecone / Weaviate):
    - Same Postgres instance → no extra infra to manage
    - JOIN between user data and vector results in one query
    - Good enough for < 10M vectors; switch to dedicated DB above that

  LangSmith (not custom logging):
    - Zero-code integration — env var enables full trace capture
    - Run tree shows every node's input/output without print statements
    - Evaluation datasets and scoring built in
    - Token cost tracking across all runs
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("3. Request lifecycle (streaming path)")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  1.  Browser opens EventSource("GET /chat/stream?message=X&thread_id=T")
  2.  FastAPI receives request
  3.  JWT middleware: verify token → extract user_id
  4.  Redis rate limiter: INCR user:user_id:req_count → check vs limit
  5.  Input guardrail: length check, content filter (harmful request?)
  6.  LangGraph: app.astream_events(input, config={thread_id: T, user_id: ...})
       a. graph runs: llm_node → tools → llm_node
       b. PostgreSQL: checkpoint written after each node
       c. OpenAI API: LLM calls (with LangSmith tracing)
       d. Tool APIs: Jira / AWS / K8s calls
  7.  FastAPI streams SSE events as they arrive:
       {"type": "tool_start", "name": "search_jira"}
       {"type": "token", "text": "I found..."}
       {"type": "done"}
  8.  Output guardrail: PII scanner runs on final answer
  9.  LangSmith: trace closed, token costs recorded
  10. Browser: EventSource.close() on "done" event
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("4. Human-in-the-loop flow (approval path)")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  1.  User sends: "Create a Jira ticket for the login bug"
  2.  LangGraph hits interrupt_before=["tools"] — graph PAUSES
  3.  FastAPI returns partial SSE stream:
        {"type": "interrupt", "pending_tools": [{"name": "create_jira_ticket", "args": {...}}]}
  4.  Browser: shows approval dialog "Create ticket PROJ-XXX?"
  5.  User clicks Approve
  6.  Browser: POST /chat/approve {"thread_id": "T", "approved": true}
  7.  FastAPI: app.invoke(None, config)  ← resume with no new input
  8.  LangGraph: tool executes → final answer streamed
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("5. Horizontal scaling considerations")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  Stateless FastAPI instances:
    - All state lives in PostgreSQL (checkpoints) and Redis (rate limits/cache)
    - Any instance can handle any request for any thread_id
    - Deploy N replicas behind a load balancer; no sticky sessions needed

  SSE connection pinning:
    - An SSE connection is long-lived (seconds to minutes)
    - The load balancer must route the same SSE request to the same instance
      OR use a message broker (Redis pub/sub, Kafka) to fan events to the
      correct instance
    - Simplest: use consistent hashing on thread_id at the LB

  Database connection pooling:
    - Each FastAPI instance uses PgBouncer or asyncpg pool (size=10)
    - At 50 instances × 10 connections = 500 pg connections max
    - PostgreSQL default max_connections=100; PgBouncer multiplexes these

  Cost model (100K requests/day, gpt-4o-mini):
    - LLM: ~$0.08/1K input tokens × avg 1K tokens = ~$8/day
    - PostgreSQL: RDS t3.medium ~$60/month
    - Redis: ElastiCache t3.micro ~$15/month
    - FastAPI: 2× t3.small ECS tasks ~$30/month
    Total: ~$100/month infrastructure + ~$240/month LLM costs
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("6. What goes wrong in production (and how to prevent it)")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  Problem                     │ Fix
  ────────────────────────────┼─────────────────────────────────────────────
  OpenAI rate limit (429)     │ Exponential backoff; use tenacity library
  LLM timeout (30s+)          │ Set timeout=30 on ChatOpenAI; stream instead
  Context window overflow     │ Summarise old messages; trim to last N turns
  Infinite agent loop         │ recursion_limit=25 in graph compile
  PII in LLM input            │ Scan + redact before sending to OpenAI
  Prompt injection            │ Separate system prompt from user content
  Cold start latency          │ Keep one warm FastAPI instance; model caching
  Checkpoint DB lock          │ Use asyncpg pool; short transaction scope
  SSE proxy buffering         │ X-Accel-Buffering: no on nginx
  Cost explosion              │ max_tokens cap + per-user daily spend limit
""")
