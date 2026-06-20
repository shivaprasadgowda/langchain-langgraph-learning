# Interview Questions — 04 Tool Calling

---

## Q1. What is tool calling and how does it work at the API level?

**Answer:**
Tool calling (also called function calling) lets an LLM request the execution of a function instead of — or in addition to — producing a text reply.

Flow:
1. You send the model a list of tool schemas (name, description, argument types).
2. The model decides whether a tool is needed.
3. If yes, it returns a structured `tool_calls` list instead of a text reply.
4. Your code executes the function with the supplied arguments.
5. You send the result back as a `ToolMessage` (section 05).
6. The model reads the result and produces a final natural-language answer.

The model **never runs code itself** — it only requests tool calls. Execution is always your responsibility.

---

## Q2. What does the `@tool` decorator do?

**Answer:**
It converts a plain Python function into a LangChain `StructuredTool` by:
- Using the **function name** as the tool name sent to the model.
- Using the **docstring** as the tool description — this is what the model reads to decide when to call the tool. A good docstring is critical.
- Deriving the **argument JSON schema** from type hints.

It also makes the tool directly `.invoke()`-able with a dict of arguments, which is how your dispatch loop calls it after receiving a `tool_call` from the model.

---

## Q3. What is `bind_tools` and what does it return?

**Answer:**
`llm.bind_tools(tools)` returns a **new Runnable** (not a mutation of the original) that includes the tool schemas in every API request it makes. The original `llm` is unchanged.

```python
llm_with_tools = llm.bind_tools([search, create_ticket])
# llm is still bare — no tools
# llm_with_tools sends tool schemas with every call
```

You can also pass `tool_choice`:
- `"auto"` (default) — model decides whether to call a tool.
- `"any"` — model must call at least one tool.
- `"none"` — tool schemas are sent but the model won't call them.
- `{"type": "function", "function": {"name": "my_tool"}}` — force a specific tool.

---

## Q4. What does `response.tool_calls` look like and what fields does each entry have?

**Answer:**
It is a list of dicts. Each entry has four keys:

```python
{
    "name": "create_jira_ticket",          # which tool to call
    "args": {"title": "Login bug", ...},   # arguments the model chose
    "id":   "call_abc123xyz",              # unique ID per call
    "type": "tool_call",                   # always "tool_call"
}
```

The `id` is essential — it must be passed to `ToolMessage(tool_call_id=...)` so the model can match the result to the original request (section 05).

---

## Q5. Can the model call multiple tools in a single response?

**Answer:**
Yes. If the user asks two independent questions, the model can return multiple entries in `tool_calls` in one response. This is called **parallel tool calling**.

```python
# "What is 10 + 5 and also 4 * 7?"
response.tool_calls == [
    {"name": "add",      "args": {"a": 10, "b": 5}, "id": "call_1", ...},
    {"name": "multiply", "args": {"a": 4,  "b": 7}, "id": "call_2", ...},
]
```

You must execute both and return two `ToolMessage` objects, one per `id`.

---

## Q6. What makes a good tool docstring?

**Answer:**
The docstring is the model's only guide for deciding whether to call the tool and what arguments to pass. A good docstring:

- States **what** the tool does in one clear sentence.
- Lists each argument with its meaning and allowed values if not obvious from type hints.
- Gives the **format** for string arguments (`YYYY-MM-DD`, region like `us-east-1`).
- Explains any preconditions (`b must not be zero`).

A vague docstring (e.g. `"""Process data."""`) causes the model to call the tool at the wrong time or with wrong arguments.

---

## Q7. What is the difference between a one-round tool call (section 04) and a full agent loop (section 05)?

**Answer:**

| | Section 04 | Section 05 |
|--|-----------|-----------|
| Rounds | One | Multiple |
| After execution | Print the result | Send a `ToolMessage` back to the model |
| Final answer | Not produced | Model reads tool result and writes a natural-language reply |
| Use case | Learning the mechanics | Real agent behaviour |

In section 04 the loop stops after dispatching the tool. In a real agent the result is fed back so the model can reason about it, potentially call more tools, and finally answer the user.

---

## Q8. Why define a `tool_map` dictionary?

**Answer:**
The model returns a tool name as a string (e.g. `"create_jira_ticket"`). You need to look up the actual Python function by that name and call it. A `dict` keyed by name makes this O(1) and avoids a long `if/elif` chain:

```python
tool_map = {t.name: t for t in tools}

# Dispatch
result = tool_map[tc["name"]].invoke(tc["args"])
```

This pattern scales cleanly as the tool list grows and is exactly what LangGraph's `ToolNode` does internally (section 10).
