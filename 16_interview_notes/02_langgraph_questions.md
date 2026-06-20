# LangGraph Interview Questions

---

## Q1. What is LangGraph and why was it created on top of LangChain?

**Answer:**
LangGraph is a library for building stateful, multi-step AI agents as directed graphs. It was created because LCEL chains are **acyclic** — they run A→B→C and stop. But real agents need **cycles**: ask the LLM, call a tool, return the result to the LLM, repeat until done.

LangGraph adds:
- **Stateful execution** — a typed `State` dict that persists across node executions
- **Cycles** — conditional edges can route back to earlier nodes
- **Persistence** — checkpointing after every node (survives restarts)
- **Human-in-the-loop** — the graph can pause and wait for human approval
- **Streaming** — token-level and node-level event streaming built in
- **Time-travel** — replay from any past checkpoint

Without LangGraph you'd write a `while True` loop with manual state management. That works for simple cases but breaks down for parallel tool calls, HITL, persistence, and debugging at scale.

---

## Q2. Walk me through the four core LangGraph primitives.

**Answer:**

**State** — a `TypedDict` that defines all fields the graph tracks. Each node receives the full state and returns a partial update.
```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent:   str
```

**Node** — any Python function that takes `State` and returns `dict`. The dict is merged into state using reducers.
```python
def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}
```

**Edge** — defines where execution goes after a node.
- Normal edge: `add_edge("node_a", "node_b")` — always goes to B after A
- Conditional edge: `add_conditional_edges("node_a", routing_fn, {route_map})` — routing_fn inspects state and returns a key

**Reducer** — specifies how to merge a node's partial update into state.
- `Annotated[list, add_messages]` — appends new messages, deduplicates by `id`
- Default (no annotation) — last-write-wins (new value replaces old)

---

## Q3. What is the `add_messages` reducer and why is it critical?

**Answer:**
`add_messages` is a reducer for the `messages` field that:
1. **Appends** new messages to the existing list (instead of replacing it)
2. **Deduplicates** by `message.id` — if you send a message with the same `id` as an existing one, the incoming message **replaces** the old one (last-write-wins on same id)

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

Without `add_messages`, each node would need to return the full message history — `{"messages": state["messages"] + [new_msg]}`. With it, nodes just return `{"messages": [new_msg]}` and the reducer handles accumulation.

The deduplication is critical for human-in-the-loop: when rejecting a pending action, you update state with a modified `AIMessage` that has the same `id` — the reducer replaces the original tool-call message without appending a duplicate.

---

## Q4. Explain `START` and `END` — what are they and why does LangGraph need them?

**Answer:**
`START` and `END` are virtual nodes that mark graph entry and exit points.

- `add_edge(START, "first_node")` — defines where the graph begins when `.invoke()` is called
- `add_edge("last_node", END)` — tells LangGraph that execution is complete; the graph returns current state

They are needed because LangGraph builds an explicit graph structure — there is no implicit "first node" or "done" signal. Multiple entry points are possible (less common), and multiple nodes can route to `END` (e.g., both success and error paths).

```python
from langgraph.graph import START, END

builder.add_edge(START, "classifier")
builder.add_conditional_edges("classifier", route_fn, {
    "jira":    "jira_node",
    "general": "general_node",
})
builder.add_edge("jira_node",    END)
builder.add_edge("general_node", END)
```

---

## Q5. How does human-in-the-loop (HITL) work in LangGraph?

**Answer:**
LangGraph HITL uses two mechanisms:

**Mechanism 1 — `interrupt_before` at compile time:**
```python
app = builder.compile(interrupt_before=["tools"])
```
When the graph reaches the `tools` node, it pauses and returns. Resume with:
```python
app.invoke(None, config)   # None = no new input, just continue
```

**Mechanism 2 — `interrupt()` inside a node:**
```python
from langgraph.types import interrupt, Command

def approval_node(state: State) -> Command:
    decision = interrupt({"question": "Approve this action?"})
    if decision == "yes":
        return Command(goto="execute_node")
    return Command(goto="cancel_node")
```
Resume with `Command(resume="yes")` passed as input to the next `.invoke()`.

**Rejection pattern** — to cancel without executing, use `update_state()` to replace the pending `AIMessage` (with `tool_calls`) with one that has empty `tool_calls`, using the same message `id`:
```python
snapshot = app.get_state(config)
bad_msg   = snapshot.values["messages"][-1]
fixed_msg = AIMessage(content="Cancelled", id=bad_msg.id)  # same id!
app.update_state(config, {"messages": [fixed_msg]}, as_node="llm")
```

---

## Q6. What is the difference between `app.stream(stream_mode="updates")` and `stream_mode="messages"`?

**Answer:**

| Mode | What it yields | When to use |
|------|---------------|-------------|
| `"updates"` | `{node_name: partial_state_update}` — one dict per node after it completes | Logging which nodes ran; progress indicators; debugging |
| `"values"` | Full accumulated state after each node | When you need complete state at every step |
| `"messages"` | `(AIMessageChunk, metadata)` tuples — individual LLM tokens with node attribution | Streaming tokens to a UI in real time |

```python
# Progress indicator: know when each node starts/ends
for event in app.stream(input, stream_mode="updates"):
    node_name = list(event.keys())[0]
    print(f"Completed: {node_name}")

# Token streaming for chat UI
for chunk, meta in app.stream(input, stream_mode="messages"):
    if chunk.type == "AIMessageChunk" and chunk.content:
        print(chunk.content, end="", flush=True)
```

---

## Q7. How do compiled sub-graphs work in multi-agent systems?

**Answer:**
A compiled LangGraph app (`app = builder.compile()`) can be added as a node in a parent graph:

```python
specialist_app = specialist_builder.compile()
parent.add_node("jira_agent", specialist_app)
```

When the parent graph executes the `"jira_agent"` node:
1. It passes the current parent state to `specialist_app.invoke(state)`
2. The specialist runs its own full agent loop (LLM → tools → LLM)
3. When the specialist reaches `END`, it returns its final state
4. The parent merges the specialist's state update into parent state via reducers

**Requirement:** parent and specialist must share compatible state schemas. At minimum, the `messages` field must exist in both and use `add_messages` so the specialist's messages are appended to the parent's history.

This pattern enables each specialist to have its own focused tools and system prompt while sharing a unified conversation history with the parent.

---

## Q8. When would you choose a manual `while True` agent loop over LangGraph?

**Answer:**
Almost never in production. But the valid cases are:

**Choose manual loop if:**
- The task is a one-shot CLI script with no persistence, no HITL, no streaming
- You want zero dependencies beyond the OpenAI SDK
- The agent logic is 10 lines and adding a graph would triple the code

**Always choose LangGraph if:**
- You need persistence (conversations survive restarts)
- You need HITL (pause and wait for approval)
- You have multiple parallel tool calls and want state management
- You need to debug via LangSmith trace trees
- You're building a multi-agent system
- You need streaming with node-level attribution
- The agent will run in a multi-instance production environment

The real cost of a manual loop is what you build *later* — adding persistence means a DB layer, adding HITL means a pause/resume protocol, adding multi-agent means composing loops. LangGraph gives you all of this for free.
