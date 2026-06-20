# 07 — LangGraph Router

> Status: **Complete**

## What This Section Covers

- Classifier node using structured output
- Conditional edges based on classification
- Routing to specialist nodes: Jira, AWS, Kubernetes, General

## Files

| File | Purpose |
|------|---------|
| `01_classifier_node.py` | LangGraph node wrapping a `with_structured_output` classifier; minimal one-node graph |
| `02_conditional_edges.py` | `add_conditional_edges` with explicit `path_map`; keyword classifier for speed; four specialist stubs |
| `03_full_router_graph.py` | Production-style full graph: LLM classifier → conditional edge → four domain-specialist LLM nodes |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
python 07_langgraph_router/01_classifier_node.py
python 07_langgraph_router/02_conditional_edges.py
python 07_langgraph_router/03_full_router_graph.py
```

## Graph Shape

```
START
  │
  ▼
classifier_node   ← structured-output LLM sets state["intent"]
  │
  ▼ (conditional edge reads state["intent"])
  ├──► jira_node
  ├──► aws_node
  ├──► kubernetes_node
  └──► general_node
           │
          END
```

## Key Concepts

**Classifier node** — a regular LangGraph node that calls `with_structured_output` and writes `intent` to state. Expensive step done once.

**Routing function** — NOT a node; read-only dict lookup on `state["intent"]`; returns a node name string. Passed to `add_conditional_edges`.

**`path_map`** — explicit `{return_value: node_name}` dict in `add_conditional_edges`; enables compile-time validation of all destinations.

**Specialist nodes** — each has a domain-specific system prompt; isolated, independently testable; easy to extend by adding one node + one edge entry.

**Scaling to multi-agent** — in section 12, each destination becomes a compiled sub-graph with its own tools and loop.
