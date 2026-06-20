# Interview Questions — 06 LangGraph Basics

---

## Q1. What is LangGraph and what problem does it solve over a manual while-loop?

**Answer:**
LangGraph is a framework for building stateful, graph-structured agent workflows on top of LangChain. It models agent logic as a directed graph where nodes are units of work and edges are flow-control decisions.

Problems it solves compared to a manual `while True` loop:

| Problem | Manual loop | LangGraph |
|---------|------------|-----------|
| Persistence | State lost on crash | Checkpointer writes state after every node |
| Branching | Linear only | Conditional edges route to different nodes |
| Human-in-the-loop | Requires complex callbacks | `interrupt()` pauses the graph cleanly |
| Streaming | Custom plumbing | Every node streams automatically |
| Testability | Monolithic function | Nodes are isolated, independently testable |

---

## Q2. What are the four core primitives of LangGraph?

**Answer:**

| Primitive | What it is |
|-----------|-----------|
| **State** | A `TypedDict` (or Pydantic model) that flows through every node. Holds all data the graph produces and consumes. |
| **Node** | A Python function `(state) → dict`. Receives full state, returns only the fields it updates. |
| **Edge** | Defines the next node. Normal edges always go to the same node; conditional edges call a function on state to decide. |
| **Reducer** | A merge function on a state field. `add_messages` appends; default is last-write-wins. |

---

## Q3. What does `Annotated[list[BaseMessage], add_messages]` mean on a state field?

**Answer:**
It attaches the `add_messages` reducer to the `messages` field. When a node returns `{"messages": [new_msg]}`, LangGraph calls `add_messages(existing_list, [new_msg])` instead of replacing the list.

Without it:
```python
messages: list[BaseMessage]   # last-write-wins — node return overwrites entire list
```

With it:
```python
messages: Annotated[list[BaseMessage], add_messages]  # new messages are appended
```

This is how multi-turn chat history accumulates across nodes without each node having to carry the full history itself.

---

## Q4. What must a node function return?

**Answer:**
A partial state update — a `dict` containing only the fields the node changed. LangGraph merges it into the current state using the registered reducers.

```python
def my_node(state: State) -> dict:
    result = do_something(state["messages"])
    return {"intent": result}   # only update 'intent'; 'messages' is unchanged
```

A node does **not** return the full state object — only the delta. Returning unchanged fields is harmless but wastes bandwidth.

---

## Q5. What is the difference between a normal edge and a conditional edge?

**Answer:**

```python
# Normal — always goes to "tools" after "llm"
graph.add_edge("llm", "tools")

# Conditional — calls route_fn(state) and uses the returned string as next node
graph.add_conditional_edges("llm", route_fn)

# With explicit mapping (documents intent, validates return values)
graph.add_conditional_edges("llm", route_fn, {
    "tools": "tools",
    "end":   END,
})
```

The routing function receives the current state and returns a node name (or `END`). It is a plain Python function — it can inspect any field, not just messages.

---

## Q6. What do `START` and `END` represent?

**Answer:**
- `START` — a virtual entry node. `graph.add_edge(START, "my_node")` means `"my_node"` is the first real node to run when `.invoke()` or `.stream()` is called.
- `END` — a virtual terminal node. When the graph reaches `END` (either via a normal edge or a conditional edge returning `END`), execution stops and the final state is returned.

A graph can have multiple paths to `END` (e.g. a successful path and an error path both end at `END`).

---

## Q7. What does `graph.compile()` do?

**Answer:**
`compile()` validates the graph (checks for disconnected nodes, missing edges, unreachable `END`) and returns an executable `Runnable` — typically called `app`. After compiling you can call:

- `app.invoke(state)` — run synchronously and return final state.
- `app.stream(state)` — yield events from each node as they complete.
- `app.astream(state)` — async version of stream.

`compile()` also accepts a `checkpointer` argument to enable persistence, and `interrupt_before`/`interrupt_after` for human-in-the-loop pauses (section 11).

---

## Q8. How does LangGraph's state update model differ from passing arguments explicitly?

**Answer:**
In a manual loop you pass data via function arguments and return values — the caller must thread every piece of data through every call. LangGraph uses a **shared state** model:

- Every node reads from the same state dict.
- Every node writes back only what it changed.
- Downstream nodes see all updates automatically.

This means a node added anywhere in the graph can access any field without changing every caller's signature. It also means state is a single source of truth that the checkpointer can snapshot atomically.

The trade-off: state fields must be defined upfront in the `TypedDict`, and any node can read any field (less encapsulation than explicit argument passing).
