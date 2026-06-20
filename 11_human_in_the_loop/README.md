# 11 — Human-in-the-Loop

> Status: **Complete**

## What This Section Covers

- The interrupt/resume pattern in LangGraph
- Pausing a graph for human approval
- Example: approve before creating a Jira ticket

## Files

| File | Purpose |
|------|---------|
| `01_interrupt_concept.py` | `interrupt_before` mechanics; pause/inspect/resume/reject lifecycle; `update_state` |
| `02_approval_flow.py` | `interrupt()` function + `Command(resume=...)`; approve and reject scenarios with structured plan |
| `03_jira_approval_example.py` | Production-style agent: `interrupt_before=["tools"]`; Jira approve, Jira reject, weather bypass |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
python 11_human_in_the_loop/01_interrupt_concept.py
python 11_human_in_the_loop/02_approval_flow.py
python 11_human_in_the_loop/03_jira_approval_example.py
```

## Interrupt / Resume Lifecycle

```
app.invoke(initial_state, config)
         │
         ▼
    [graph runs...]
         │
    interrupt hit   ← interrupt_before=["node"] or interrupt() inside node
         │
    invoke() returns early (graph is PAUSED, not done)
         │
    app.get_state(config)  ← inspect what the agent wants to do
         │
    ┌────┴────────────────┐
    │ Approve             │ Reject
    │ invoke(None,config) │ update_state(config, ...) then invoke(None,config)
    └────┬────────────────┘
         │
    [graph resumes from pause point]
         │
        END
```

## Key Concepts

**Checkpointer required** — interrupt persists state between `invoke()` calls; without a checkpointer there is nowhere to save the pause point.

**Two interrupt methods:**
- `interrupt_before=["tools"]` at compile time — always pauses before the named node.
- `interrupt()` inside a node — conditional pause; human response is returned as the function's return value; resume with `Command(resume=value)`.

**Inspect paused state** — `app.get_state(config).next` shows pending nodes; `.values["messages"][-1].tool_calls` shows what the agent wants to run.

**Reject pattern** — call `app.update_state(config, new_msg_with_same_id)` to replace the tool-calling AIMessage before resuming; `add_messages` deduplicates by `id`.

**Production** — state lives in PostgreSQL between HTTP requests; approve and reject can hit different server replicas.
