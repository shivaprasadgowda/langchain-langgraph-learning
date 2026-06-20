# 10 — ToolNode Agents

> Status: **Complete**

## What This Section Covers

- `ToolNode` — the managed alternative to a manual loop
- `tools_condition` for conditional edges
- Full agent loop with LangGraph
- Comparison: ToolNode vs manual loop

## Files

| File | Purpose |
|------|---------|
| `01_toolnode_basics.py` | `ToolNode` standalone: single call, parallel calls, unknown tool error, `tools_by_name` |
| `02_tools_condition.py` | `tools_condition` routing demo; graph with cycle; direct routing function test |
| `03_agent_graph.py` | Production-style agent: system message, `MemorySaver`, multi-turn, parallel tool calls |
| `04_manual_vs_toolnode.py` | Same question run through both approaches; printed comparison table |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
python 10_toolnode_agents/01_toolnode_basics.py
python 10_toolnode_agents/02_tools_condition.py
python 10_toolnode_agents/03_agent_graph.py
python 10_toolnode_agents/04_manual_vs_toolnode.py
```

## Graph Shape

```
START → llm_node ──(tool_calls?)──► tools_node ──┐
            ▲                                      │
            └──────────────────────────────────────┘
            │
         (no tool_calls)
            │
           END
```

## Key Concepts

**`ToolNode(tools)`** — replaces `tool_map`, dispatch loop, `ToolMessage` construction, and parallel call handling. One line.

**`tools_condition`** — pre-built routing fn: `tool_calls present → "tools"`, else `END`. Replaces the manual `if not response.tool_calls: break`.

**`add_edge("tools", "llm")`** — creates the agent cycle. No `while` loop needed.

**System message in `llm_node`** — prepend at call time, not in state, to guarantee it's always present and not duplicated in checkpoints.

**`recursion_limit`** — set in config at invoke time; replaces `max_steps` guard; raises `GraphRecursionError` when hit.

**`interrupt_before=["tools"]`** — pause before `ToolNode` executes for human approval (section 11).
