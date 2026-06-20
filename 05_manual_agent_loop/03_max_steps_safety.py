"""
Concept: max_steps safety limit and why manual loops break at scale.

Without a cap, a buggy tool or a confused model can loop forever, burning
tokens and money. A max_steps guard is the minimum safety net for any
agent loop — even when you trust the model.

This file also demonstrates the three main failure modes that make manual
loops painful at scale and explains what LangGraph solves.

Run:
    python 05_manual_agent_loop/03_max_steps_safety.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()


# ── Tools (one intentionally always returns an ambiguous result) ──────────────

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def check_status(service: str) -> str:
    """Check the status of a service. Returns 'unknown' if not found."""
    # Always returns 'unknown' — the model may keep re-trying
    return "unknown — service not found in registry"


tools = [add, check_status]
tool_map = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


# ── Agent loop with max_steps guard ──────────────────────────────────────────

class MaxStepsReached(Exception):
    pass


def run_agent(user_input: str, max_steps: int = 5) -> str:
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=user_input),
    ]

    for step in range(1, max_steps + 1):
        print(f"  [step {step}/{max_steps}] calling model...")
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"  [step {step}] done — model gave final answer")
            return response.content

        print(f"  [step {step}] {len(response.tool_calls)} tool call(s):")
        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            print(f"           → {tc['name']}({tc['args']}) = {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # Reached the limit — raise rather than silently returning garbage
    raise MaxStepsReached(
        f"Agent did not finish within {max_steps} steps. "
        f"Last tool calls: {[tc['name'] for tc in response.tool_calls]}"
    )


# ── Demo 1: normal completion ─────────────────────────────────────────────────
print("=" * 60)
print("Demo 1 — normal completion")
print("=" * 60)
result = run_agent("What is 99 plus 1?", max_steps=5)
print("Answer:", result)


# ── Demo 2: max_steps triggered ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Demo 2 — max_steps triggered (service always returns 'unknown')")
print("=" * 60)
try:
    run_agent(
        "Keep checking the status of 'ghost-service' until it shows as 'running'.",
        max_steps=3,
    )
except MaxStepsReached as e:
    print(f"Caught: {e}")


# ── Why manual loops become hard at scale ─────────────────────────────────────
print("\n" + "=" * 60)
print("Why manual loops break at scale")
print("=" * 60)

problems = [
    (
        "No persistence",
        "The history list lives in memory. If the process restarts mid-loop "
        "(network error, deploy, crash) all context is lost. "
        "LangGraph checkpointers write state after every step.",
    ),
    (
        "No branching",
        "A while-loop is linear. Real agents need conditional paths: "
        "route to Jira node OR AWS node based on intent. "
        "LangGraph edges model this as a graph.",
    ),
    (
        "No human-in-the-loop",
        "Pausing for approval mid-loop means blocking the thread or using "
        "complex async/callback patterns. "
        "LangGraph's interrupt() suspends the graph cleanly.",
    ),
    (
        "No streaming",
        "Streaming intermediate steps to a UI requires passing callbacks "
        "deep into the loop. LangGraph streams events from every node "
        "automatically.",
    ),
    (
        "Hard to test",
        "The entire loop is one function. LangGraph nodes are isolated "
        "functions — each one can be unit-tested independently.",
    ),
]

for title, explanation in problems:
    print(f"\n  Problem: {title}")
    print(f"  {explanation}")
