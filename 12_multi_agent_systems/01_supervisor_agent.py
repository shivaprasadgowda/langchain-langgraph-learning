"""
Concept: Supervisor agent — routes tasks to the right specialist.

The supervisor is the entry point of a multi-agent system. It does NOT
solve the task itself. Its only job is to:
  1. Read the user's request.
  2. Decide which specialist is best suited to handle it.
  3. Hand the request off (via a conditional edge).

The supervisor uses structured output (section 03 pattern) so the routing
decision is a typed Python object, not a fragile string parse.

This file shows the supervisor in isolation — section 03 wires it into the
full system with specialist sub-agents.

Run:
    python 12_multi_agent_systems/01_supervisor_agent.py
"""

from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


# ── Shared state (used by all agents in this section) ─────────────────────────

class State(TypedDict):
    messages:    Annotated[list[BaseMessage], add_messages]
    next_agent:  str   # set by supervisor, read by graph router
    final_answer: str  # set by whichever specialist handles the request


# ── Routing schema ────────────────────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    """The supervisor's routing decision."""

    next: Literal["jira_agent", "aws_agent", "kubernetes_agent", "general_agent"] = Field(
        description=(
            "jira_agent       — Jira tickets, sprints, epics, or project tracking.\n"
            "aws_agent        — AWS services: EC2, S3, Lambda, RDS, IAM, costs.\n"
            "kubernetes_agent — pods, deployments, services, Helm, kubectl.\n"
            "general_agent    — everything else."
        )
    )
    reasoning: str = Field(description="One sentence explaining the routing choice.")


# ── Supervisor node ───────────────────────────────────────────────────────────

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_supervisor_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a supervisor that routes user requests to the correct specialist agent. "
            "Analyse the request and choose the most appropriate agent.",
        ),
        ("human", "{request}"),
    ])
    | _llm.with_structured_output(SupervisorDecision)
)


def supervisor_node(state: State) -> dict:
    """Classify the request and set next_agent in state."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"), ""
    )
    decision: SupervisorDecision = _supervisor_chain.invoke({"request": last_human})
    print(f"  [supervisor] → {decision.next!r}  ({decision.reasoning})")
    return {"next_agent": decision.next}


# ── Router function (reads state, returns node name) ─────────────────────────

def route_to_agent(state: State) -> str:
    return state["next_agent"]


# ── Stub specialist nodes (just to show routing works standalone) ─────────────

def jira_stub(state: State) -> dict:
    return {"final_answer": f"[JIRA STUB] handled: {state['messages'][-1].content[:50]}"}

def aws_stub(state: State) -> dict:
    return {"final_answer": f"[AWS STUB] handled: {state['messages'][-1].content[:50]}"}

def kubernetes_stub(state: State) -> dict:
    return {"final_answer": f"[K8S STUB] handled: {state['messages'][-1].content[:50]}"}

def general_stub(state: State) -> dict:
    return {"final_answer": f"[GENERAL STUB] handled: {state['messages'][-1].content[:50]}"}


# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(State)
builder.add_node("supervisor",       supervisor_node)
builder.add_node("jira_agent",       jira_stub)
builder.add_node("aws_agent",        aws_stub)
builder.add_node("kubernetes_agent", kubernetes_stub)
builder.add_node("general_agent",    general_stub)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "jira_agent":       "jira_agent",
        "aws_agent":        "aws_agent",
        "kubernetes_agent": "kubernetes_agent",
        "general_agent":    "general_agent",
    },
)

for agent in ["jira_agent", "aws_agent", "kubernetes_agent", "general_agent"]:
    builder.add_edge(agent, END)

app = builder.compile()


# ── Test routing ──────────────────────────────────────────────────────────────

test_requests = [
    "Create a high-priority Jira ticket for the auth service 401 errors.",
    "Why is our RDS bill so high this month?",
    "The payments pod is OOMKilled — how do I increase its memory limit?",
    "What is the difference between SQL and NoSQL databases?",
]

print(f"{'Request':<60} {'Routed to'}")
print("─" * 85)

for req in test_requests:
    result = app.invoke({
        "messages":     [HumanMessage(content=req)],
        "next_agent":   "",
        "final_answer": "",
    })
    print(f"{req[:58]:<60} {result['next_agent']}")
