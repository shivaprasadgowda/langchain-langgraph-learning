"""
Concept: Router-ready structured output.

A router node in LangGraph reads the classification result and calls
add_conditional_edges() to decide which node runs next.

For this to work cleanly:
  - The output must be a typed object (not a string).
  - The routing field should use Literal[...] so only valid destinations exist.
  - The router function just reads one field — no string parsing needed.

This file shows the full pattern:
  user message → classifier chain → RouteDecision → router function → next node

Section 07 wires this directly into a LangGraph graph.

Run:
    python 03_structured_output/05_router_output.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


# ── Schema ────────────────────────────────────────────────────────────────────
# The `destination` field maps directly to LangGraph node names (section 07).

class RouteDecision(BaseModel):
    """Routing decision for a DevOps support chatbot."""

    destination: Literal["jira_node", "aws_node", "kubernetes_node", "general_node"] = Field(
        description=(
            "jira_node       — Jira tickets, sprints, or project management.\n"
            "aws_node        — AWS services, costs, IAM, EC2, S3, Lambda.\n"
            "kubernetes_node — pods, deployments, services, Helm, kubectl.\n"
            "general_node    — everything else."
        )
    )
    reasoning: str = Field(description="One sentence explaining the routing choice.")


# ── Classifier chain ──────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You route user messages to the correct specialist node in a "
        "DevOps support system. Choose the most specific node that applies.",
    ),
    ("human", "{message}"),
])

classifier = template | llm.with_structured_output(RouteDecision)


# ── Router function ───────────────────────────────────────────────────────────
# In LangGraph this function is passed to add_conditional_edges().
# It receives state and returns the node name to jump to.
# Here we simulate it by passing the RouteDecision directly.

def router(decision: RouteDecision) -> str:
    """Return the destination node name. Called by LangGraph conditional edge."""
    return decision.destination


# ── Specialist node stubs (would be real LangGraph nodes in section 07) ───────

def jira_node(message: str) -> str:
    return f"[JIRA NODE] Handling: {message[:50]}"

def aws_node(message: str) -> str:
    return f"[AWS NODE] Handling: {message[:50]}"

def kubernetes_node(message: str) -> str:
    return f"[K8S NODE] Handling: {message[:50]}"

def general_node(message: str) -> str:
    return f"[GENERAL NODE] Handling: {message[:50]}"

NODE_MAP = {
    "jira_node": jira_node,
    "aws_node": aws_node,
    "kubernetes_node": kubernetes_node,
    "general_node": general_node,
}


# ── Full routing pipeline ─────────────────────────────────────────────────────

def handle(message: str) -> None:
    decision: RouteDecision = classifier.invoke({"message": message})
    next_node = router(decision)
    output = NODE_MAP[next_node](message)

    print(f"Message   : {message[:60]}")
    print(f"→ Route   : {decision.destination}")
    print(f"  Reason  : {decision.reasoning}")
    print(f"  Output  : {output}")
    print()


test_messages = [
    "The staging-api pod is in CrashLoopBackOff — can you check the logs?",
    "Create a bug ticket for the payment timeout issue.",
    "Our S3 bucket costs jumped 40% — what's going on?",
    "What does SOLID stand for?",
]

for msg in test_messages:
    handle(msg)
