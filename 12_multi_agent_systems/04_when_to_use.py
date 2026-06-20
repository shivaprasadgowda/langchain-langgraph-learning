"""
Concept: When to use multi-agent — and when not to.

Multi-agent adds orchestration overhead. It is only worth the complexity
when the problem genuinely requires it. This file is a decision guide —
no API calls needed.

Run:
    python 12_multi_agent_systems/04_when_to_use.py
"""


def section(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


section("USE multi-agent WHEN...")
print("""
  1. DOMAIN SPECIALISATION MATTERS
     Different domains need different tools, prompts, and expertise.
     A Jira agent needs Jira tools. A Kubernetes agent needs kubectl tools.
     One prompt trying to be expert in both dilutes quality in both.
     → Use specialist agents, each with focused tooling.

  2. CONTEXT WINDOW MANAGEMENT
     Each specialist starts with a clean context containing only its own
     conversation. A single agent accumulating 40 tool calls across Jira,
     AWS, and Kubernetes risks hitting the context limit and loses precision.
     → Specialists have shorter, more focused histories.

  3. PARALLEL WORKLOADS
     LangGraph can fan out to multiple specialists simultaneously (e.g.
     check AWS cost AND create a Jira ticket at the same time).
     A single agent must do these sequentially.
     → Multi-agent enables real parallelism.

  4. INDEPENDENT DEPLOYABILITY
     In production you may want the Jira agent to use a different model
     or be deployed in a different region from the AWS agent.
     Compiled sub-graphs are independent Runnables — they can be
     containerised separately.

  5. INDEPENDENT TESTABILITY
     Each specialist can be unit-tested with a focused set of inputs.
     A monolithic agent test matrix grows combinatorially.
""")


section("DO NOT use multi-agent WHEN...")
print("""
  1. THE PROBLEM IS SIMPLE
     If one agent with 3-5 tools handles everything, adding a supervisor
     just adds latency and cost. A ToolNode agent (section 10) is faster,
     simpler, and easier to debug.

  2. TASKS SHARE STATE DEEPLY
     If specialist A's output must be a direct input to specialist B's
     first tool call, the handoff adds friction. A single agent with both
     toolsets avoids the state-passing overhead.

  3. YOU HAVE A SMALL CORPUS OF TOOLS (< 8)
     Modern LLMs handle 8-10 tools in one prompt without confusion.
     The routing step in a multi-agent system is an extra LLM call
     (cost + latency) that only pays off when specialisation improves quality.

  4. LATENCY IS CRITICAL
     Each agent transition is a round-trip to the LLM. A 3-hop multi-agent
     chain (supervisor → specialist → tools) adds ~1-2s versus a single
     agent doing the same work. For real-time chat this is often unacceptable.

  5. YOU ARE STILL PROTOTYPING
     Start with a single ToolNode agent. Extract to multi-agent only when
     you can measure that specialisation improves quality or that context
     window limits are genuinely a problem.
""")


section("Decision flowchart")
print("""
  Do you have 2+ clearly distinct domains with domain-specific tools?
       │
       ├─ No  → Single ToolNode agent (section 10)
       │
       └─ Yes
           │
           Do tasks ever need results from multiple domains in one reply?
           │
           ├─ No  → Simple router graph (section 07) — one specialist per turn
           │
           └─ Yes
               │
               Do you need parallelism or independent scaling per domain?
               │
               ├─ No  → Router with specialist nodes (section 07 pattern)
               │
               └─ Yes → Multi-agent with supervisor + sub-graphs (this section)
""")


section("Cost of adding a supervisor")
print("""
  Every request makes an extra LLM call to the supervisor.
  At $0.15 / 1M input tokens (gpt-4o-mini), for a short classification
  prompt (~200 tokens in, ~50 out):

    Cost per classification ≈ $0.000045

  That is negligible at low volume but adds up at 1M requests/day:
    ~$45/day just for routing — before the specialist calls.

  Mitigation:
    - Use a small, fast model (gpt-4o-mini or even a local model) for the supervisor.
    - Cache routing decisions for identical or near-identical inputs.
    - Use keyword routing first (no LLM) and fall back to LLM only for ambiguous cases.
""")


section("Summary table")
print("""
  ┌─────────────────────────────┬──────────────────┬────────────────────┐
  │ Factor                      │ Single agent     │ Multi-agent        │
  ├─────────────────────────────┼──────────────────┼────────────────────┤
  │ Number of tool domains      │ 1–2              │ 3+                 │
  │ Context window pressure     │ Low              │ High               │
  │ LLM calls per request       │ N                │ N + 1 (supervisor) │
  │ Latency                     │ Lower            │ Higher             │
  │ Specialist prompt quality   │ Shared / generic │ Focused / expert   │
  │ Independent testing         │ Hard             │ Easy               │
  │ Parallel execution          │ No               │ Yes (fan-out)      │
  │ Complexity                  │ Low              │ High               │
  └─────────────────────────────┴──────────────────┴────────────────────┘
""")
