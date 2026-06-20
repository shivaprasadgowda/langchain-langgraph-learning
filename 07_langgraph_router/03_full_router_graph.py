"""
Concept: End-to-end router graph — LLM classifier + specialist nodes.

This is the full production-style pattern combining:
  - Section 03: with_structured_output for reliable classification
  - Section 06: StateGraph, nodes, conditional edges
  - Section 07: classifier node + specialist nodes

Graph shape:
    START
      │
      ▼
  classifier_node  (structured-output LLM classification)
      │
      ▼ (conditional edge — reads state["intent"])
   ┌──┴──────────────────────┐
   │         │               │              │
   ▼         ▼               ▼              ▼
jira_node  aws_node  kubernetes_node  general_node
   │         │               │              │
   └─────────┴───────────────┴──────────────┘
                             │
                            END

Each specialist node uses a system prompt tailored to its domain.
The final answer is always in state["messages"][-1].

Run:
    python 07_langgraph_router/03_full_router_graph.py
"""

from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════════════

class State(TypedDict):
    messages:   Annotated[list[BaseMessage], add_messages]
    intent:     str
    confidence: str


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER — structured output LLM
# ═══════════════════════════════════════════════════════════════════════════════

class Classification(BaseModel):
    """Intent classification for a DevOps support chatbot."""

    intent: Literal["jira", "aws", "kubernetes", "general"] = Field(
        description=(
            "jira        — Jira tickets, sprints, epics, boards, or project tracking.\n"
            "aws         — AWS services, EC2, S3, Lambda, RDS, IAM, costs, or CloudWatch.\n"
            "kubernetes  — pods, deployments, services, ingress, Helm, namespaces, or kubectl.\n"
            "general     — everything else including architecture, concepts, and coding."
        )
    )
    confidence: Literal["high", "medium", "low"]


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_classifier_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an intent classifier for a DevOps support chatbot. "
            "Classify the user message accurately.",
        ),
        ("human", "{message}"),
    ])
    | _llm.with_structured_output(Classification)
)


def classifier_node(state: State) -> dict:
    """Classify the most recent human message and store intent in state."""
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"), None
    )
    if last_human is None:
        return {"intent": "general", "confidence": "low"}

    result: Classification = _classifier_chain.invoke({"message": last_human.content})
    print(f"  [classifier] {result.intent!r} ({result.confidence})")
    return {"intent": result.intent, "confidence": result.confidence}


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALIST NODES — each uses a domain-specific system prompt
# ═══════════════════════════════════════════════════════════════════════════════

def _specialist_response(state: State, system_prompt: str) -> dict:
    """Shared helper: prepend a system message and call the LLM."""
    response = _llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}


def jira_node(state: State) -> dict:
    """Handle Jira-related requests."""
    return _specialist_response(
        state,
        "You are a Jira expert. Help the user with tickets, sprints, epics, "
        "and project management. Be concise and practical.",
    )


def aws_node(state: State) -> dict:
    """Handle AWS-related requests."""
    return _specialist_response(
        state,
        "You are an AWS solutions architect. Help with EC2, S3, Lambda, RDS, "
        "IAM, costs, and best practices. Be concise and practical.",
    )


def kubernetes_node(state: State) -> dict:
    """Handle Kubernetes-related requests."""
    return _specialist_response(
        state,
        "You are a Kubernetes expert. Help with pods, deployments, services, "
        "Helm, namespaces, and kubectl commands. Be concise and practical.",
    )


def general_node(state: State) -> dict:
    """Handle general questions."""
    return _specialist_response(
        state,
        "You are a helpful DevOps assistant. Answer the user's question clearly "
        "and concisely.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def route(state: State) -> str:
    """Read state["intent"] and return the destination node name."""
    return {
        "jira":       "jira_node",
        "aws":        "aws_node",
        "kubernetes": "kubernetes_node",
        "general":    "general_node",
    }.get(state["intent"], "general_node")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

builder = StateGraph(State)

builder.add_node("classifier",      classifier_node)
builder.add_node("jira_node",       jira_node)
builder.add_node("aws_node",        aws_node)
builder.add_node("kubernetes_node", kubernetes_node)
builder.add_node("general_node",    general_node)

builder.add_edge(START, "classifier")

builder.add_conditional_edges(
    "classifier",
    route,
    {
        "jira_node":       "jira_node",
        "aws_node":        "aws_node",
        "kubernetes_node": "kubernetes_node",
        "general_node":    "general_node",
    },
)

for node in ["jira_node", "aws_node", "kubernetes_node", "general_node"]:
    builder.add_edge(node, END)

app = builder.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def ask(question: str) -> None:
    print(f"\n{'─'*65}")
    print(f"User: {question}")
    result = app.invoke({
        "messages":   [HumanMessage(content=question)],
        "intent":     "",
        "confidence": "",
    })
    answer = result["messages"][-1].content
    print(f"[{result['intent'].upper()}] {answer[:200]}")


ask("Create a critical Jira ticket: prod login page returns 500 since last deploy.")
ask("Our Lambda cold start times jumped from 200ms to 2s — how do I fix this?")
ask("The payments-service pod is OOMKilled. How do I increase its memory limit?")
ask("What is the difference between horizontal and vertical scaling?")
