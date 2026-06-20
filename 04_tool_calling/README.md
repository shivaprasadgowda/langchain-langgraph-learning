# 04 — Tool Calling

> Status: **Complete**

## What This Section Covers

- Defining tools with `@tool`
- Binding tools to the model with `bind_tools`
- Reading `response.tool_calls`
- Example tools: calculator, dummy Jira, AWS, Bitbucket

## Files

| File | Purpose |
|------|---------|
| `01_define_tool.py` | `@tool` decorator — name, description, args schema, direct `.invoke()` |
| `02_bind_tools.py` | `bind_tools`, `tool_choice`, when the model does/doesn't call a tool |
| `03_read_tool_calls.py` | Parse `response.tool_calls`; dispatch single and parallel calls |
| `04_calculator_tool.py` | add/subtract/multiply/divide/power tools; one-round dispatch |
| `05_dummy_integrations.py` | Jira, AWS, Bitbucket stub tools; multi-tool agent demo |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
python 04_tool_calling/01_define_tool.py    # no API call needed
python 04_tool_calling/02_bind_tools.py
python 04_tool_calling/03_read_tool_calls.py
python 04_tool_calling/04_calculator_tool.py
python 04_tool_calling/05_dummy_integrations.py
```

## Key Concepts

**`@tool`** — wraps a function; docstring = what the model reads to decide when to call it; type hints = argument schema.

**`bind_tools(tools)`** — returns a new Runnable that includes tool schemas in every request. Original `llm` unchanged.

**`response.tool_calls`** — list of `{name, args, id, type}` dicts. Empty if the model answered directly.

**`tool_map`** — `{t.name: t for t in tools}` — O(1) lookup from tool name string to callable.

**One-round vs agent loop** — section 04 dispatches but doesn't feed results back. Section 05 closes the loop with `ToolMessage`.
