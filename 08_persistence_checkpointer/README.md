# 08 — Persistence & Checkpointer

> Status: **Complete**

## What This Section Covers

- `add_messages` reducer for chat history
- `InMemorySaver` checkpointer
- `thread_id` for isolated conversations
- Running multiple conversations in parallel
- Concept: PostgreSQL checkpointer for production

## Files

| File | Purpose |
|------|---------|
| `01_add_messages_reducer.py` | Reducer mechanics: append, deduplication by id, last-write-wins comparison, `RemoveMessage` |
| `02_in_memory_saver.py` | Wire `MemorySaver` to a graph; multi-turn memory without manual history; `get_state()` |
| `03_thread_id_isolation.py` | Three threads (alice, bob, ticket) with full isolation verification |
| `04_postgres_concept.py` | No DB needed — prints patterns for `AsyncPostgresSaver`, pooling, FastAPI lifespan, SQLite, and time-travel |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
python 08_persistence_checkpointer/01_add_messages_reducer.py   # no API key needed
python 08_persistence_checkpointer/02_in_memory_saver.py
python 08_persistence_checkpointer/03_thread_id_isolation.py
python 08_persistence_checkpointer/04_postgres_concept.py       # no API key needed
```

## Key Concepts

**`add_messages` reducer** — appends new messages instead of replacing; deduplicates by `id`; supports `RemoveMessage`.

**`MemorySaver`** — in-process dict; zero setup; lost on restart. Dev/test only.

**`thread_id`** — the conversation key in `config["configurable"]["thread_id"]`; one app serves unlimited users.

**`compile(checkpointer=...)`** — only this line changes between dev (`MemorySaver`) and prod (`AsyncPostgresSaver`).

**Crash recovery** — checkpoint written after every node; failed graph resumes from last completed node, not from `START`.

**Time-travel** — `get_state_history(config)` lists all checkpoints; re-invoke with a past `checkpoint_id` to branch history.
