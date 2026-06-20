"""
Concept: add_conditional_edges — how LangGraph routing works.

add_conditional_edges(source_node, routing_fn, path_map)

  source_node — the node whose output triggers the routing decision.
  routing_fn  — a function (state) -> str that returns the next node name.
  path_map    — optional dict {return_value: node_name}.
                Makes all possible destinations explicit and allows
                LangGraph to validate them at compile time.

The routing function is NOT a node — it does not update state.
It only reads state and returns a destination string.

Run:
    python 07_langgraph_router/02_conditional_edges.py
"""

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages:   Annotated[list[BaseMessage], add_messages]
    intent:     str
    route_taken: str   # for observability — which path was chosen


# ── Nodes ─────────────────────────────────────────────────────────────────────

def classifier_node(state: State) -> dict:
    """Simple keyword classifier (no LLM — keeps this example fast)."""
    last = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
    text = last.content.lower() if last else ""

    if any(w in text for w in ["jira", "ticket", "sprint", "epic"]):
        intent = "jira"
    elif any(w in text for w in ["aws", "ec2", "s3", "lambda", "iam", "cost"]):
        intent = "aws"
    elif any(w in text for w in ["pod", "kubernetes", "kubectl", "helm", "deployment"]):
        intent = "kubernetes"
    else:
        intent = "general"

    return {"intent": intent}


def jira_node(state: State) -> dict:
    response = llm.invoke(state["messages"] + [HumanMessage(
        content="Answer this Jira-related question concisely."
    )])
    return {"messages": [response], "route_taken": "jira"}


def aws_node(state: State) -> dict:
    response = llm.invoke(state["messages"] + [HumanMessage(
        content="Answer this AWS-related question concisely."
    )])
    return {"messages": [response], "route_taken": "aws"}


def kubernetes_node(state: State) -> dict:
    response = llm.invoke(state["messages"] + [HumanMessage(
        content="Answer this Kubernetes-related question concisely."
    )])
    return {"messages": [response], "route_taken": "kubernetes"}


def general_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response], "route_taken": "general"}


# ── Routing function ──────────────────────────────────────────────────────────
# This is NOT a node. It reads state and returns a destination string.
# LangGraph calls it automatically after classifier_node completes.

def route_by_intent(state: State) -> str:
    """Map intent value to destination node name."""
    mapping = {
        "jira":       "jira_node",
        "aws":        "aws_node",
        "kubernetes": "kubernetes_node",
        "general":    "general_node",
    }
    destination = mapping.get(state["intent"], "general_node")
    print(f"  [router] intent={state['intent']!r} → {destination!r}")
    return destination


# ── Build graph ───────────────────────────────────────────────────────────────

builder = StateGraph(State)

builder.add_node("classifier",       classifier_node)
builder.add_node("jira_node",        jira_node)
builder.add_node("aws_node",         aws_node)
builder.add_node("kubernetes_node",  kubernetes_node)
builder.add_node("general_node",     general_node)

builder.add_edge(START, "classifier")

# Conditional edge: after classifier runs, call route_by_intent to pick next node.
# path_map makes every possible destination explicit — LangGraph validates these.
builder.add_conditional_edges(
    "classifier",
    route_by_intent,
    {
        "jira_node":       "jira_node",
        "aws_node":        "aws_node",
        "kubernetes_node": "kubernetes_node",
        "general_node":    "general_node",
    },
)

# All specialist nodes end the graph
builder.add_edge("jira_node",       END)
builder.add_edge("aws_node",        END)
builder.add_edge("kubernetes_node", END)
builder.add_edge("general_node",    END)

app = builder.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

queries = [
    "Create a high-priority Jira ticket for the payment timeout.",
    "What EC2 instance type is best for a memory-intensive workload?",
    "How do I scale a deployment to 5 replicas in Kubernetes?",
    "Explain the CAP theorem.",
]

for q in queries:
    print(f"\nUser: {q}")
    result = app.invoke({
        "messages":    [HumanMessage(content=q)],
        "intent":      "",
        "route_taken": "",
    })
    print(f"Route: {result['route_taken']}")
    print(f"Answer: {result['messages'][-1].content[:120]}")
