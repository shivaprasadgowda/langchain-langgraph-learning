# Interview Questions — 07 LangGraph Router

---

## Q1. What is the router pattern and why is it useful in agent systems?

**Answer:**
The router pattern uses a classifier node to read the user's intent and a conditional edge to direct execution to the most appropriate specialist node. Instead of one large prompt trying to handle everything, each specialist node has a focused system prompt and toolset.

Benefits:
- **Better answers** — a Kubernetes-focused prompt outperforms a generic one on k8s questions.
- **Separation of concerns** — each specialist node can be developed, tested, and updated independently.
- **Observability** — `state["intent"]` is a concrete, loggable routing decision.
- **Scalability** — adding a new domain means adding a node and an edge entry, not rewriting the core loop.

---

## Q2. What is the difference between the classifier node and the routing function?

**Answer:**

| | Classifier node | Routing function |
|--|----------------|-----------------|
| Is it a node? | Yes — registered with `add_node` | No — passed to `add_conditional_edges` |
| Updates state? | Yes — writes `intent`, `confidence` | No — read-only |
| Called by LangGraph? | Executed as a step in the graph | Called internally after the source node |
| Returns | A partial state dict `{"intent": ...}` | A node name string `"jira_node"` |

The classifier does the expensive work (LLM call). The routing function is a trivial dict lookup on the result.

---

## Q3. Why use `with_structured_output` in the classifier node instead of parsing a text reply?

**Answer:**
Parsing free text is fragile — the model might return `"Jira"`, `"JIRA"`, `"jira ticket"`, or wrap it in a sentence. Any mismatch breaks the routing logic.

`with_structured_output(Classification)` forces the model to return `intent` as one of the exact `Literal` values (`"jira"`, `"aws"`, `"kubernetes"`, `"general"`). The routing function is then a guaranteed-safe dict lookup. No `.lower()`, no `.strip()`, no guard clauses.

---

## Q4. What is the `path_map` argument to `add_conditional_edges` and why include it?

**Answer:**
```python
builder.add_conditional_edges(
    "classifier",
    route,
    {                              # ← path_map
        "jira_node":       "jira_node",
        "aws_node":        "aws_node",
        "kubernetes_node": "kubernetes_node",
        "general_node":    "general_node",
    },
)
```

The `path_map` explicitly lists every possible destination. LangGraph uses it to:
1. **Validate** at compile time that all destinations are registered nodes.
2. **Document** all possible routes for graph visualization tools.

Without it, LangGraph accepts any string the routing function returns and only fails at runtime if the string doesn't match a node name.

---

## Q5. Why does the routing function not update state?

**Answer:**
The routing function is a pure control-flow decision — it answers "where do we go next?" without changing any data. Mixing routing logic with state mutation makes the graph harder to reason about: you'd need to read two places (the node and the routing function) to understand what state looks like after a step.

LangGraph enforces this separation by design: `add_conditional_edges` accepts a function, not a node. Only functions registered with `add_node` can update state.

---

## Q6. How would you add a new "database" intent to this router?

**Answer:**
Three additions only:

```python
# 1. Add the new literal to the Classification schema
intent: Literal["jira", "aws", "kubernetes", "database", "general"]

# 2. Add a specialist node
def database_node(state: State) -> dict:
    return _specialist_response(state, "You are a database expert...")

builder.add_node("database_node", database_node)

# 3. Add the destination to the path_map and the routing dict
# In route():
    "database": "database_node"
# In add_conditional_edges path_map:
    "database_node": "database_node"
# And an edge to END:
builder.add_edge("database_node", END)
```

The classifier, the conditional edge mechanism, and every other node are untouched.

---

## Q7. What happens if the classifier returns an intent not in the routing dict?

**Answer:**
With the `Literal` constraint in the Pydantic schema, the model can only return one of the defined values — Pydantic raises a `ValidationError` otherwise. So under normal operation this cannot happen.

As a defensive measure, the routing function uses `.get(state["intent"], "general_node")` so an unexpected value falls through to the general handler rather than raising a `KeyError`. This is a belt-and-suspenders guard for cases where the schema evolves.

---

## Q8. How does this router pattern scale to a multi-agent system (section 12)?

**Answer:**
The router in this section routes to simple LLM nodes. In section 12, each destination is itself a sub-graph (a specialist agent with its own tools, memory, and loop):

```
classifier_node
      │
      ├──► jira_agent       (sub-graph: LLM + Jira tools + loop)
      ├──► aws_agent        (sub-graph: LLM + AWS tools + loop)
      ├──► kubernetes_agent (sub-graph: LLM + k8s tools + loop)
      └──► general_agent    (sub-graph: LLM + RAG + loop)
```

The routing logic is identical — only the destination changes from a single node to a compiled sub-graph. LangGraph supports this via `add_node("jira_agent", jira_app)` where `jira_app` is itself a compiled `StateGraph`.
