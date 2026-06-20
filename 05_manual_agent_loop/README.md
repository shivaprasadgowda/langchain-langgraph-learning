# 05 — Manual Agent Loop

> Status: **Complete**

## What This Section Covers

- `ToolMessage` and feeding results back to the LLM
- Building a manual agent loop from scratch
- Adding a `max_steps` safety limit
- Why manual loops become hard to manage at scale

## Files

| File | Purpose |
|------|---------|
| `01_tool_message.py` | Construct `ToolMessage`, step-by-step two-call flow with history trace |
| `02_manual_loop.py` | Full while-loop agent: single tool, multi-step, parallel tool calls |
| `03_max_steps_safety.py` | `MaxStepsReached` guard; demo of a runaway loop; five scaling problems |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
python 05_manual_agent_loop/01_tool_message.py
python 05_manual_agent_loop/02_manual_loop.py
python 05_manual_agent_loop/03_max_steps_safety.py
```

## Key Concepts

**`ToolMessage`** — carries tool result back to model; requires `content` (str) and `tool_call_id` (from `tc["id"]`).

**Message order** — `SystemMessage → HumanMessage → AIMessage (tool_calls) → ToolMessage(s) → AIMessage (final answer)`. The `AIMessage` with tool_calls *must* appear in history before its `ToolMessage`.

**Loop exit condition** — `response.tool_calls == []` means the model has its final answer.

**`max_steps` guard** — minimum safety net for any agent loop; LangGraph uses `recursion_limit` for the same purpose.

**Why manual loops break at scale** — no persistence, no branching, no human-in-the-loop, no streaming, hard to test. LangGraph solves all five.
