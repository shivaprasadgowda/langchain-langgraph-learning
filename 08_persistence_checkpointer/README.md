# 08 — Persistence & Checkpointer

> Status: **TODO**

## What This Section Covers

- `add_messages` reducer for chat history
- `InMemorySaver` checkpointer
- `thread_id` for isolated conversations
- Running multiple conversations in parallel
- Concept: PostgreSQL checkpointer for production

## Files (to be created)

| File | Purpose |
|------|---------|
| `01_add_messages_reducer.py` | Demonstrate the reducer |
| `02_in_memory_saver.py` | Wire up InMemorySaver |
| `03_thread_id_isolation.py` | Two threads, separate history |
| `04_postgres_concept.py` | Code sketch + comments only |
| `interview_questions.md` | Interview Q&A for this section |
