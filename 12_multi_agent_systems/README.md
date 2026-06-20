# 12 — Multi-Agent Systems

> Status: **Complete**

## What This Section Covers

- Supervisor agent pattern
- Specialist sub-agents
- When to use multi-agent vs a single agent
- When NOT to use multi-agent

## Files

| File | Purpose |
|------|---------|
| `01_supervisor_agent.py` | Supervisor node with structured-output routing; stub destinations; routing table test |
| `02_specialist_agents.py` | Four compiled specialist sub-graphs (Jira, AWS, K8s, General) tested independently |
| `03_full_multi_agent_graph.py` | Full system: supervisor → conditional edge → specialist sub-graph → response_node |
| `04_when_to_use.py` | Decision flowchart, cost analysis, summary table — no API calls |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
python 12_multi_agent_systems/01_supervisor_agent.py
python 12_multi_agent_systems/02_specialist_agents.py
python 12_multi_agent_systems/03_full_multi_agent_graph.py
python 12_multi_agent_systems/04_when_to_use.py     # no API key needed
```

## Architecture

```
START
  │
  ▼
supervisor_node  ← structured-output LLM classifies intent → sets next_agent
  │
  ▼ (conditional edge reads next_agent)
  ├──► jira_agent        (compiled sub-graph: LLM + Jira tools)
  ├──► aws_agent         (compiled sub-graph: LLM + AWS tools)
  ├──► kubernetes_agent  (compiled sub-graph: LLM + K8s tools)
  └──► general_agent     (compiled sub-graph: LLM only)
           │
        response_node  (extracts final answer from messages)
           │
          END
```

## Key Concepts

**Supervisor** — classifies intent with structured output; sets `next_agent`; does NOT solve the task.

**Specialist as compiled sub-graph** — `build.compile()` → add with `parent.add_node("name", compiled_app)`.

**State compatibility** — parent and sub-graph must share the same field names/types; `add_messages` reducer merges sub-graph messages into parent history.

**When to use** — 3+ distinct domains, context window pressure, parallel workloads, independent scaling needs.

**When NOT to use** — simple tasks, < 8 tools, latency-critical, early prototype stage.
