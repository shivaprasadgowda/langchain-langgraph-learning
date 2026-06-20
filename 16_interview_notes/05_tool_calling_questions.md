# Tool Calling Interview Questions

---

## Q1. How does tool calling work at the API level?

**Answer:**
Tool calling is a structured way for the LLM to request that the application execute a function and return the result. The full cycle:

1. **Define tools** — send a JSON Schema description of each function to the API
2. **LLM responds** — instead of text, the model returns a `tool_calls` array: `[{name: "get_weather", id: "call_xyz", args: {city: "Tokyo"}}]`
3. **Execute** — your code calls the real function with those args
4. **Return result** — send a `ToolMessage(content=result, tool_call_id="call_xyz")` back to the LLM
5. **LLM generates final answer** — with the tool result in context

```python
# Step 2 — check if model wants to call tools
response = llm.invoke(messages)
if response.tool_calls:
    for tc in response.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    # Step 5 — get final answer
    final = llm.invoke(messages)
```

The critical detail: `tool_call_id` links the `ToolMessage` back to the specific `tool_calls` entry. Without matching IDs, the model doesn't know which tool result belongs to which call.

---

## Q2. What is the `@tool` decorator and what does it generate?

**Answer:**
`@tool` turns a Python function into a LangChain `Tool` object with three key attributes:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 22°C"
```

- `get_weather.name` → `"get_weather"` (function name)
- `get_weather.description` → `"Get the current weather for a city."` (docstring)
- `get_weather.args_schema` → Pydantic schema auto-generated from type hints

The docstring is the most important field — the LLM reads it to decide when to call the tool. A bad description leads to wrong tool selection.

You can override with explicit metadata:
```python
@tool(name="weather_api", description="Fetch real-time weather data for any city worldwide.")
def get_weather(city: str) -> str: ...
```

---

## Q3. What is `bind_tools()` and what does it do to the LLM?

**Answer:**
`bind_tools()` attaches tool definitions to an LLM instance so every call to that LLM includes the tool schemas:

```python
tools = [search_jira, get_aws_cost, create_ticket]
llm_with_tools = llm.bind_tools(tools)
```

Under the hood it sets `tools` in the API request body:
```json
{
  "model": "gpt-4o-mini",
  "messages": [...],
  "tools": [
    {"type": "function", "function": {"name": "search_jira", "description": "...", "parameters": {...}}},
    ...
  ]
}
```

The LLM uses tool descriptions to decide which tool (if any) to call. You can also force a specific tool:
```python
llm.bind_tools(tools, tool_choice="search_jira")      # always call this tool
llm.bind_tools(tools, tool_choice="any")              # must call at least one tool
llm.bind_tools(tools, tool_choice="none")             # don't call any tools
```

---

## Q4. How do you handle parallel tool calls?

**Answer:**
Modern LLMs can request multiple tools in a single response. `response.tool_calls` is a list — iterate and execute all before feeding results back:

```python
response = llm.invoke(messages)
messages.append(response)  # always append the AIMessage first

if response.tool_calls:
    for tc in response.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(
            content      = str(result),
            tool_call_id = tc["id"],   # must match the specific tool_calls entry
        ))
    # Now feed ALL results back together
    final = llm.invoke(messages)
```

**Common mistake:** appending ToolMessages one-by-one and calling the LLM after each. You must collect **all** ToolMessages for a given turn and send them together — the API requires one ToolMessage per tool_call in the AIMessage.

ToolNode handles this correctly automatically:
```python
builder.add_node("tools", ToolNode(tools))
```

---

## Q5. What is `ToolNode` and how does it differ from a manual tool dispatch loop?

**Answer:**

| | Manual dispatch | `ToolNode` |
|--|----------------|-----------|
| Tool dispatch | `tool_map[tc["name"]].invoke(tc["args"])` | Automatic |
| Parallel calls | Must iterate `response.tool_calls` correctly | Automatic |
| ToolMessage construction | Manual | Automatic (correct `tool_call_id`) |
| Unknown tool | `KeyError` uncaught | Raises `KeyError` (same) |
| Error handling | Custom | Returns error as `ToolMessage` content |
| LangSmith tracing | Manual `@traceable` | Automatic per-tool spans |

`ToolNode` is the LangGraph-native tool executor. It reads the last `AIMessage`'s `tool_calls`, executes each tool, and returns `{"messages": [ToolMessage, ToolMessage, ...]}`.

```python
from langgraph.prebuilt import ToolNode, tools_condition

builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("llm", tools_condition)  # routes to "tools" or END
builder.add_edge("tools", "llm")                       # always loop back
```

---

## Q6. What is `tools_condition` and what does it check?

**Answer:**
`tools_condition` is a built-in routing function that inspects the last message and returns `"tools"` if tool calls are present, or `END` if not:

```python
from langgraph.prebuilt import tools_condition

def tools_condition(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END
```

It's used as the routing function in `add_conditional_edges`:
```python
builder.add_conditional_edges("llm", tools_condition)
```

This creates the classic agent loop: the LLM runs, if it wants tools we execute them and loop back, if it's done (no tool calls) we exit.

---

## Q7. How do you design good tool descriptions for reliable tool selection?

**Answer:**
The LLM reads the `description` field to decide whether to call a tool. Good descriptions follow these rules:

**Be specific about what the tool does and doesn't do:**
```python
# Bad
@tool
def jira(query: str) -> str:
    """Jira tool."""

# Good
@tool
def search_jira_tickets(query: str) -> str:
    """Search open Jira tickets by keyword. Use for finding existing bugs,
    tasks, or feature requests. Does NOT create new tickets."""
```

**Include when to use vs when NOT to use:**
Especially important when you have similar tools (e.g., `search_jira` vs `create_jira_ticket`).

**Be explicit about argument format:**
```python
@tool
def get_aws_cost(month: str) -> str:
    """Get AWS cost summary. month must be in YYYY-MM format (e.g., '2026-06')."""
```

**Avoid tool name collisions** in meaning — if two tools have similar descriptions, the model will pick the wrong one unpredictably.

---

## Q8. How would you build a tool that requires human approval before executing?

**Answer:**
Two approaches in LangGraph:

**Approach A — `interrupt_before=["tools"]`** (section 11):
```python
app = builder.compile(interrupt_before=["tools"])
# Graph pauses before ToolNode runs
# Call app.get_state() to inspect pending tool_calls
# Resume with app.invoke(None, config) to approve
# Or update_state() to inject a cancelled AIMessage to reject
```

**Approach B — approval node with `interrupt()`**:
```python
def approval_node(state: State) -> Command:
    pending = state["messages"][-1].tool_calls
    decision = interrupt({
        "question": f"Approve execution of {[tc['name'] for tc in pending]}?",
        "tools":    pending,
    })
    if decision == "yes":
        return Command(goto="tools")
    return Command(goto="cancelled")
```

**Which to use:**
- `interrupt_before` is simpler and applies to all tool calls automatically
- `interrupt()` gives you fine-grained control (approve some tool calls but not others, or only interrupt for destructive tools)
