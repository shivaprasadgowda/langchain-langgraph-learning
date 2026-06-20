# Interview Questions — 11 Human-in-the-Loop

---

## Q1. What is human-in-the-loop (HITL) in the context of LangGraph?

**Answer:**
HITL is the pattern of pausing a running graph at a defined point, surfacing the agent's proposed action to a human for review, and then either resuming or redirecting based on that review.

Use cases:
- Approve before irreversible actions (delete, deploy, send email)
- Review tool call arguments before execution
- Provide additional information the agent could not retrieve automatically
- Override or correct the agent's plan mid-execution

LangGraph supports this natively because every pause is backed by a checkpointer snapshot — the graph state is fully preserved until the human responds, even if hours pass.

---

## Q2. What are the two ways to interrupt a graph in LangGraph?

**Answer:**

**Method A — `interrupt_before` / `interrupt_after` (compile-time):**
```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],   # pause before ToolNode runs
)
```
Always pauses at that node for every execution. Good for approval gates on all tool calls.

**Method B — `interrupt()` function (node-time, modern preferred):**
```python
from langgraph.types import interrupt, Command

def approval_node(state):
    response = interrupt("Please review: " + state["plan"])
    return {"approved": response == "yes"}
```
Pauses conditionally from inside a node. More flexible — the node decides whether to interrupt based on state (e.g. only interrupt for high-risk actions).

Both require a checkpointer.

---

## Q3. Why is a checkpointer required for interrupt to work?

**Answer:**
When a graph is interrupted, the process returns control to the caller — the `invoke()` call returns early. The full graph state (all messages, all fields set so far) must be preserved somewhere so the graph can resume later.

A checkpointer writes this snapshot to storage (memory, SQLite, PostgreSQL). Without it, the state would be lost the moment `invoke()` returns and resumption would be impossible.

This is also why HITL works across HTTP requests in a web application — the state lives in the database, not in process memory.

---

## Q4. How do you resume after an interrupt?

**Answer:**

**Method A (interrupt_before):**
```python
# Resume with no changes — equivalent to "approve"
app.invoke(None, config=config)
```

**Method B (interrupt() function):**
```python
from langgraph.types import Command

# Pass the human's response back to the interrupt() call
app.invoke(Command(resume="yes"), config=config)
```

In both cases you must use the same `config` (same `thread_id`) so LangGraph loads the correct checkpoint.

---

## Q5. How do you reject (cancel) a proposed action?

**Answer:**
You modify state before resuming. The typical pattern for `interrupt_before=["tools"]` is to replace the last `AIMessage` (which contains tool_calls) with one that has no tool_calls:

```python
snapshot = app.get_state(config)
last_ai  = snapshot.values["messages"][-1]

app.update_state(
    config,
    {"messages": [AIMessage(
        content="Action rejected by human reviewer. No changes were made.",
        id=last_ai.id,   # same id causes add_messages to replace it
    )]},
)
app.invoke(None, config=config)   # resume — tools_condition now sees no tool_calls → END
```

Setting the same `id` on the new `AIMessage` is critical — `add_messages` uses `id` for deduplication, so the old tool-calling message is replaced rather than appended.

---

## Q6. How do you inspect what the agent wants to do while the graph is paused?

**Answer:**
```python
snapshot = app.get_state(config)

# Which nodes are pending?
print(snapshot.next)          # e.g. ('tools',)

# What is the last message? (the AIMessage with tool_calls)
last_msg = snapshot.values["messages"][-1]
for tc in last_msg.tool_calls:
    print(tc["name"], tc["args"])
```

`snapshot.next` tells you which node would run next on resume. `snapshot.values` gives you the full state at the point of interruption. This is what you would display to the human in a review UI.

---

## Q7. What is the difference between `interrupt_before` and `interrupt_after`?

**Answer:**

| | `interrupt_before=["node"]` | `interrupt_after=["node"]` |
|--|---|---|
| When it pauses | Before the node runs | After the node runs |
| Node has executed? | No | Yes |
| Use case | Review before action | Review the result before continuing |
| State at pause | Node's inputs only | Includes node's outputs |

`interrupt_before=["tools"]` is the standard pattern for approval gates — you pause before the tool executes so the human can prevent it.

`interrupt_after=["llm"]` is useful when you want a human to review or edit the model's response before it is shown to the end user.

---

## Q8. How does HITL work in a production web application?

**Answer:**
In a FastAPI app backed by PostgreSQL:

```python
# Turn 1 — user sends message, agent plans, graph pauses
@app.post("/chat")
async def chat(message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    await graph.ainvoke({"messages": [HumanMessage(message)]}, config=config)
    snapshot = await graph.aget_state(config)
    if snapshot.next:
        pending_tools = snapshot.values["messages"][-1].tool_calls
        return {"status": "awaiting_approval", "pending": pending_tools}
    return {"status": "done", "reply": snapshot.values["messages"][-1].content}

# Turn 2 — human approves or rejects
@app.post("/approve")
async def approve(thread_id: str, approved: bool):
    config = {"configurable": {"thread_id": thread_id}}
    if approved:
        result = await graph.ainvoke(None, config=config)
    else:
        # update state to remove tool_calls, then resume
        ...
    return {"reply": result["messages"][-1].content}
```

The state is durably persisted in PostgreSQL between the two HTTP requests, so the two endpoints can run on different server replicas.
