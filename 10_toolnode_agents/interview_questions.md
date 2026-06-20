# Interview Questions — 10 ToolNode Agents

---

## Q1. What is `ToolNode` and what does it replace from the manual agent loop?

**Answer:**
`ToolNode` is a pre-built LangGraph node that replaces the entire tool-dispatch boilerplate from the manual loop:

| Manual loop (section 05) | ToolNode |
|--------------------------|----------|
| `tool_map = {t.name: t for t in tools}` | Built into `ToolNode` |
| `for tc in response.tool_calls:` | Internal |
| `result = tool_map[tc["name"]].invoke(tc["args"])` | Internal |
| `ToolMessage(content=str(result), tool_call_id=tc["id"])` | Internal |
| Parallel calls need extra handling | Runs parallel calls concurrently by default |

```python
# Replaces ~10 lines of dispatch code:
tool_node = ToolNode(tools)
builder.add_node("tools", tool_node)
```

---

## Q2. What does `tools_condition` do and what does it return?

**Answer:**
`tools_condition` is a pre-built routing function for `add_conditional_edges`. It inspects the last message in `state["messages"]`:

- If the last `AIMessage` has non-empty `tool_calls` → returns `"tools"` (route to ToolNode).
- Otherwise → returns `END` (model is done, exit the graph).

```python
builder.add_conditional_edges("llm", tools_condition)
```

This replaces the manual `if not response.tool_calls: break` check.

---

## Q3. What creates the agent loop cycle in the graph?

**Answer:**
Two edges together create the cycle:

```python
builder.add_conditional_edges("llm", tools_condition)  # llm → tools or END
builder.add_edge("tools", "llm")                        # tools → llm (always)
```

After `ToolNode` runs and appends `ToolMessage`s to state, the graph routes back to `llm_node`. The LLM reads the tool results and either calls more tools (another cycle) or produces its final answer (routes to `END`).

There is no explicit `while` loop — LangGraph's graph execution engine drives the cycle.

---

## Q4. How does `ToolNode` handle parallel tool calls?

**Answer:**
When the LLM requests multiple tools in one `AIMessage` (multiple entries in `tool_calls`), `ToolNode` executes them concurrently using `asyncio` or thread-based parallelism internally, then collects all results into a list of `ToolMessage` objects — one per `tool_call_id`.

```python
# Model requests two tools at once
tool_calls = [
    {"name": "add",      "args": {"a": 10, "b": 5}, "id": "c1"},
    {"name": "multiply", "args": {"a": 4,  "b": 7}, "id": "c2"},
]
# ToolNode returns:
{"messages": [
    ToolMessage(content="15",  tool_call_id="c1"),
    ToolMessage(content="28",  tool_call_id="c2"),
]}
```

In the manual loop you had to iterate manually; `ToolNode` handles this without extra code.

---

## Q5. Why should the system message be injected in `llm_node` rather than the initial state?

**Answer:**
The system message should always be the first message the LLM sees. If it is stored in `state["messages"]` with `add_messages`, it will be in the history permanently — but if someone sends an `invoke()` without it, the agent loses its persona.

The safer pattern is to prepend it programmatically in `llm_node`:

```python
def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke([SYSTEM] + state["messages"])]}
```

This guarantees the system message is always present, regardless of what the caller puts in the initial state. It is also not persisted in the checkpoint — reducing storage and preventing duplicate system messages across turns.

---

## Q6. How do you add a recursion limit to a ToolNode agent?

**Answer:**
Pass `recursion_limit` in the config at invoke time (default is 25):

```python
config = {
    "configurable": {"thread_id": "abc"},
    "recursion_limit": 10,   # max graph steps (nodes executed)
}
app.invoke({"messages": [HumanMessage(...)]}, config=config)
```

When the limit is reached, LangGraph raises `GraphRecursionError`. This is the LangGraph equivalent of `max_steps` in the manual loop — a safety net against infinite cycles.

Note: each node execution counts as one step, so an agent that does 3 tool-call rounds uses approximately 7 steps (start → llm → tools → llm → tools → llm → tools → llm).

---

## Q7. What is the difference between `ToolNode` and `tool_node.invoke(state)` directly?

**Answer:**
`tool_node.invoke(state)` runs the node standalone for testing or debugging, outside of any graph. It takes a state dict and returns the tool results synchronously.

Inside a graph, `ToolNode` is called by LangGraph's execution engine which also:
- Writes a checkpoint after the node completes.
- Streams the node's output if `.stream()` is used.
- Handles async execution if `.ainvoke()` / `.astream()` is used.

Direct `.invoke()` is useful for unit-testing: you can construct a fake `AIMessage` with tool_calls and verify `ToolNode` dispatches and formats the results correctly — without needing a full graph.

---

## Q8. How does the ToolNode agent extend to support human-in-the-loop (section 11)?

**Answer:**
Add `interrupt_before=["tools"]` to `compile()`:

```python
app = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],   # pause before ToolNode runs
)
```

The graph pauses after `llm_node` (when tool_calls are present) and before `ToolNode` executes. At this point you can:
1. Inspect `app.get_state(config)` to see which tools the model wants to call.
2. Approve (call `app.invoke(None, config=config)` to resume) or reject (modify state and resume).

This is how approval workflows are built — the agent proposes an action, a human reviews it, and only then does it execute. Section 11 covers this in full.
