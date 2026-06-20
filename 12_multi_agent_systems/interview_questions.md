# Interview Questions — 12 Multi-Agent Systems

---

## Q1. What is the supervisor pattern and how does it differ from a simple router (section 07)?

**Answer:**

| | Simple router (section 07) | Supervisor (section 12) |
|--|---------------------------|------------------------|
| Destination | A single LLM node with a tailored prompt | A compiled sub-graph with its own tools and agent loop |
| Specialist capability | One LLM call, no tools | Full ToolNode agent loop — can call tools, multi-step |
| Context isolation | Shared state throughout | Each specialist has a focused sub-state |
| Complexity | Low | Higher — each specialist is a graph |

The supervisor makes the same routing decision (structured output → conditional edge), but each destination is a fully autonomous agent capable of multi-step tool use, not just a prompt switch.

---

## Q2. How do you add a compiled sub-graph as a node in a parent graph?

**Answer:**
```python
# Compile the sub-graph normally
jira_app = jira_builder.compile()

# Add it as a node in the parent graph
parent = StateGraph(ParentState)
parent.add_node("jira_agent", jira_app)
```

LangGraph calls `jira_app.invoke(state)` when that node runs. The sub-graph receives the parent state and returns a partial state update, which LangGraph merges back using the parent's reducers.

**Requirement:** both graphs must share compatible state schemas — at minimum, the same field names and types for any fields the sub-graph reads or writes.

---

## Q3. How does state flow between the parent graph and a specialist sub-graph?

**Answer:**
1. The parent graph passes its full current state to the sub-graph node.
2. The sub-graph runs its own agent loop (LLM → tools → LLM → ...) and accumulates messages via `add_messages`.
3. When the sub-graph reaches `END`, it returns its final state.
4. LangGraph merges the sub-graph's state update into the parent state using the parent's reducers — new messages are appended via `add_messages`.

The parent state after the specialist runs contains all messages from both the parent conversation and the specialist's tool calls. This is the foundation for the `response_node` that reads the specialist's last AI message.

---

## Q4. What are three concrete benefits of specialist agents over a single agent?

**Answer:**

1. **Focused system prompts** — a Kubernetes specialist prompt can contain detailed kubectl patterns and troubleshooting steps that would dilute a general prompt and confuse the model on non-K8s tasks.

2. **Focused toolsets** — the Kubernetes specialist only receives K8s tools. With 15 total tools across three domains, a single agent would have a cluttered tool list and risk choosing the wrong tool. Each specialist works with 3-5 relevant tools.

3. **Context window isolation** — each specialist starts a fresh, short conversation containing only the relevant exchange. A single agent accumulating 30+ tool calls across Jira, AWS, and Kubernetes approaches context limits and degrades in quality.

---

## Q5. When should you NOT use multi-agent?

**Answer:**
- **Simple problems** — one agent with 3-5 tools handles everything; a supervisor adds unnecessary latency.
- **Deep state sharing** — if specialist A's raw output is specialist B's first input, the state-passing overhead adds friction that a single agent avoids.
- **Fewer than 8 tools total** — modern LLMs handle this without confusion; specialisation doesn't improve quality enough to justify the extra routing call.
- **Latency-sensitive paths** — each supervisor round-trip is an extra 300-800ms LLM call.
- **Early prototyping** — start single-agent; extract to multi-agent only once you observe real quality or context-limit problems.

---

## Q6. What is the cost of adding a supervisor and how do you mitigate it?

**Answer:**
Every request incurs an extra LLM call (the supervisor classification). At low volume this is negligible. At 1M requests/day with `gpt-4o-mini` (a ~200-token classification prompt) it adds ~$45/day before specialist costs.

Mitigations:
- Use the smallest capable model for the supervisor (not GPT-4o).
- Use keyword routing first: if the message contains "kubectl" or "pod", skip the LLM classification and route directly. Only call the LLM for ambiguous inputs.
- Cache routing decisions for identical or near-identical queries.

---

## Q7. How does multi-agent enable parallel execution?

**Answer:**
LangGraph supports `Send` — a mechanism to fan out to multiple nodes simultaneously:

```python
from langgraph.types import Send

def supervisor_fan_out(state: State) -> list[Send]:
    # Run both specialists in parallel for a compound request
    return [
        Send("jira_agent", state),
        Send("aws_agent",  state),
    ]

parent.add_conditional_edges("supervisor", supervisor_fan_out)
```

This is more advanced than the section-12 pattern (sequential routing) but shows the key advantage: tasks that are independent of each other (create a Jira ticket AND check AWS cost) can run simultaneously, halving the latency for compound requests.

---

## Q8. How does the multi-agent pattern scale to a production system?

**Answer:**
In production, each specialist sub-graph can be:

1. **Deployed independently** — containerised and run on separate infrastructure, with its own scaling policy and resource limits.

2. **Model-differentiated** — the Jira agent might use `gpt-4o-mini` (cheap, good at structured tasks), while the Kubernetes agent uses a larger model for complex debugging.

3. **Separately monitored** — each specialist has its own LangSmith traces, error rates, and latency metrics, making it easy to identify which domain is underperforming.

4. **Independently updated** — updating the Kubernetes agent's prompt or tools doesn't risk breaking the Jira agent. Specialists can be versioned and rolled back independently.

The supervisor remains the stable entry point; specialists evolve at their own pace.
