# Interview Questions — 08 Persistence & Checkpointer

---

## Q1. What is a LangGraph checkpointer and what problem does it solve?

**Answer:**
A checkpointer is a storage backend that snapshots the full graph state after every node execution. It solves two problems:

1. **Memory** — without a checkpointer, every `app.invoke()` starts with blank state. The chatbot has no recollection of prior turns. With a checkpointer, the graph loads the previous state on each call — the caller does not manage a history list.

2. **Durability** — without a checkpointer, a process crash or deploy wipes all conversation state. With a persistent checkpointer (PostgreSQL, SQLite), conversations survive restarts, and a graph can resume from the last completed node after a failure.

---

## Q2. What is the `thread_id` and how does it enable multi-user support?

**Answer:**
`thread_id` is the key that identifies a single conversation. It is passed in `config`:

```python
config = {"configurable": {"thread_id": "user-123-session-7"}}
app.invoke(input, config=config)
```

Every `invoke()` with the same `thread_id` on the same compiled `app` shares one checkpoint history. Different `thread_id`s are completely independent — their states never bleed into each other.

One compiled `app` + one `MemorySaver` can serve thousands of users simultaneously; each user just gets their own `thread_id`.

---

## Q3. What does the `add_messages` reducer do and why is it needed?

**Answer:**
`add_messages` is a merge function attached to the `messages` field:

```python
messages: Annotated[list[BaseMessage], add_messages]
```

Without it, every node return replaces the entire list (last-write-wins). With it, LangGraph calls `add_messages(existing, new)` which **appends** new messages rather than overwriting.

It also **deduplicates by message `id`** — if the same `id` appears in both lists, the incoming version wins. This prevents duplicate messages when a checkpointer replays state on resume.

---

## Q4. What is the difference between `MemorySaver` and `AsyncPostgresSaver`?

**Answer:**

| | `MemorySaver` | `AsyncPostgresSaver` |
|--|--------------|---------------------|
| Storage | Python dict in RAM | PostgreSQL table |
| Survives restart | No | Yes |
| Multi-replica safe | No (each process has its own dict) | Yes (shared DB) |
| Setup | Zero — just import | `pip install langgraph-checkpoint-postgres` + DB |
| Use case | Dev, testing, unit tests | Production |

The graph code is identical — only the `compile(checkpointer=...)` line changes.

---

## Q5. How does a checkpointer enable crash recovery mid-graph?

**Answer:**
LangGraph writes a checkpoint after **every node**, not just at the end. Each checkpoint records which nodes have completed.

If the process crashes mid-graph (say, after node A but before node B), the next `invoke()` with the same `thread_id`:
1. Loads the latest checkpoint (state after node A).
2. Sees that node B has not yet run.
3. Resumes from node B without re-running node A.

Without a checkpointer, the entire graph would restart from `START`, potentially re-executing expensive or side-effectful nodes.

---

## Q6. What does `app.get_state(config)` return?

**Answer:**
A `StateSnapshot` object with:
- `.values` — the full state dict at the latest checkpoint.
- `.next` — the node(s) that would run next (empty if graph is finished).
- `.config` — the config used, including `thread_id` and `checkpoint_id`.
- `.metadata` — step count, node that last ran, timestamps.

```python
snapshot = app.get_state({"configurable": {"thread_id": "abc"}})
print(snapshot.values["messages"])   # full message history
print(snapshot.next)                  # e.g. ("llm",) or ()
```

`get_state_history(config)` returns all checkpoints for a thread in reverse chronological order — useful for debugging and time-travel.

---

## Q7. What is time-travel in LangGraph?

**Answer:**
Because every checkpoint is stored with a `checkpoint_id`, you can re-invoke the graph from any past point:

```python
# List checkpoints
for cp in app.get_state_history(config):
    print(cp.config["configurable"]["checkpoint_id"], cp.metadata["step"])

# Resume from a specific earlier checkpoint
old_config = {
    "configurable": {
        "thread_id":     "abc",
        "checkpoint_id": "old-checkpoint-id",
    }
}
app.invoke({"messages": [HumanMessage("take a different path")]}, config=old_config)
```

Use cases: debugging a bad agent decision, A/B testing different responses on the same history, audit trails.

---

## Q8. Why use a connection pool with `AsyncPostgresSaver` in production?

**Answer:**
Opening a new PostgreSQL connection per request is expensive (~50–100ms). A connection pool keeps a set of connections open and reuses them:

```python
from psycopg_pool import AsyncConnectionPool

pool = AsyncConnectionPool(conninfo=DB_URL, max_size=20)
checkpointer = AsyncPostgresSaver(pool)
```

With `max_size=20`, up to 20 concurrent graph executions can checkpoint simultaneously without waiting for a connection. The pool is created once at application startup (e.g. in FastAPI's `lifespan`) and closed on shutdown.

Without pooling, a burst of 50 simultaneous requests would open 50 connections, potentially exhausting the PostgreSQL connection limit and introducing significant latency.
