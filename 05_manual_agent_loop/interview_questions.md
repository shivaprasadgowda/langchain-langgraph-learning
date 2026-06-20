# Interview Questions — 05 Manual Agent Loop

---

## Q1. What is a ToolMessage and why is it required?

**Answer:**
A `ToolMessage` carries the result of a tool execution back to the model. It is required because the model's tool_call is a *request* — your code runs the tool and must report the result so the model can reason about it and produce a grounded final answer.

Without a `ToolMessage` after a tool_call, the OpenAI API raises an error — you cannot skip straight from an `AIMessage` with tool_calls to another `HumanMessage`.

Key fields:
- `content` — the tool's return value as a string.
- `tool_call_id` — copied from `tc["id"]` in `response.tool_calls`. This is how the model matches each result to the request that produced it.

---

## Q2. What is the correct message order in a tool-calling conversation?

**Answer:**

```
SystemMessage        # persona / instructions
HumanMessage         # user turn
AIMessage            # model requests tool(s) — content is often empty
ToolMessage          # result for tool_call_id = "call_abc"
ToolMessage          # result for tool_call_id = "call_def"  (if parallel)
AIMessage            # model's final natural-language answer
```

Rules:
- The `AIMessage` with `tool_calls` **must** appear in history before its `ToolMessage`(s).
- Every `tool_call_id` in `tool_calls` must have a matching `ToolMessage` before the next model call.
- Only after all tool results are present does the model produce its final text reply.

---

## Q3. What does the manual agent loop look like in code?

**Answer:**

```python
while True:
    response = llm.invoke(messages)
    messages.append(response)             # always save the AIMessage

    if not response.tool_calls:           # model finished
        return response.content

    for tc in response.tool_calls:        # execute each tool
        result = tool_map[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
```

The loop exits only when `tool_calls` is empty, which signals the model is satisfied with the information it has.

---

## Q4. Why must you append the AIMessage to history even when it has tool_calls?

**Answer:**
The model's conversation history must be a complete, contiguous record. If you skip the `AIMessage` and go straight to `ToolMessage`, the model receives a `ToolMessage` with no preceding tool request — which violates the protocol and causes an API error.

The `AIMessage` is also the record of *which* tools were requested and with *which* arguments, so the `ToolMessage`'s `tool_call_id` can be validated against it.

---

## Q5. What happens if you don't add a max_steps limit?

**Answer:**
The loop can run indefinitely if:
- A tool consistently returns an ambiguous or unhelpful result and the model keeps retrying.
- The model enters a reasoning loop where it calls tools in a cycle.
- A bug in your tool raises an exception that gets swallowed.

Without a cap you burn tokens (and money) with no progress. A `max_steps` guard is the minimum safety net. In LangGraph, `recursion_limit` serves the same role at the graph level.

---

## Q6. What are the five reasons manual loops break at scale?

**Answer:**

| Problem | Impact | LangGraph solution |
|---------|--------|--------------------|
| No persistence | Crash/restart loses all history | Checkpointer writes state after every step |
| No branching | Can't route to different handlers | Conditional edges on a graph |
| No human-in-the-loop | Blocking thread or complex callbacks | `interrupt()` suspends graph cleanly |
| No streaming | Hard to emit intermediate steps to UI | Every node's output streams automatically |
| Hard to test | Entire loop is one monolithic function | Nodes are isolated functions, independently testable |

---

## Q7. How does the model signal that it is done calling tools?

**Answer:**
When `response.tool_calls` is an empty list (`[]`), the model has all the information it needs and its `response.content` contains the final answer. That is the exit condition for the loop.

There is no special "done" message type — the absence of tool calls is the signal.

---

## Q8. What is the relationship between the manual loop in this section and LangGraph's ToolNode?

**Answer:**
`ToolNode` (section 10) is a pre-built LangGraph node that implements exactly the dispatch logic from the manual loop:

```python
# What ToolNode does internally (simplified):
def tool_node(state):
    ai_message = state["messages"][-1]
    tool_messages = []
    for tc in ai_message.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": tool_messages}
```

`tools_condition` then checks whether the last message has tool_calls (route to `ToolNode`) or not (route to `END`). Building the loop manually first makes it clear that `ToolNode` is not magic — it is just the same pattern, packaged as a reusable graph node.
