# 06 — LangGraph Basics

> Status: **Complete**

## What This Section Covers

- What is LangGraph and why it exists
- `StateGraph`, state, nodes, edges
- `START` and `END` constants
- Building and running a first graph

## Files

| File | Purpose |
|------|---------|
| `01_what_is_langgraph.py` | No API calls — prints a structured explanation of LangGraph concepts |
| `02_first_graph.py` | Minimal `START → llm_node → END` graph; `.invoke()` and `.stream()` |
| `03_state_node_edge.py` | Each primitive isolated: reducer demo, node partial-update, conditional edge, two-node graph |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
python 06_langgraph_basics/01_what_is_langgraph.py   # no API key needed
python 06_langgraph_basics/02_first_graph.py
python 06_langgraph_basics/03_state_node_edge.py
```

## Key Concepts

**State** — `TypedDict` shared across all nodes; nodes return partial updates only.

**`add_messages` reducer** — `Annotated[list[BaseMessage], add_messages]` appends new messages instead of replacing the list.

**Node** — `(state: State) -> dict`; returns only the fields it changed.

**Normal edge** — `add_edge("a", "b")` — always goes to `b`.

**Conditional edge** — `add_conditional_edges("a", fn)` — calls `fn(state)` and routes to the returned node name.

**`START` / `END`** — virtual entry and terminal nodes; a graph can have multiple paths to `END`.

**`compile()`** — validates graph and returns an executable `Runnable` with `.invoke()` / `.stream()`.
