"""
Concept: Specialist agents — compiled sub-graphs with focused tools and prompts.

Each specialist is a fully independent compiled StateGraph:
  - Its own system prompt tailored to its domain
  - Its own tool set (only tools it needs)
  - Its own LLM node + ToolNode agent loop

Because each specialist is a compiled Runnable, it can be:
  - Added as a node in the parent graph (section 03)
  - Tested independently (just call specialist.invoke(...))
  - Deployed as its own service in future (sub-graph isolation)

This file builds and tests each specialist in isolation.

Run:
    python 12_multi_agent_systems/02_specialist_agents.py
"""

from dotenv import load_dotenv
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Shared state schema ───────────────────────────────────────────────────────

class State(TypedDict):
    messages:    Annotated[list[BaseMessage], add_messages]
    next_agent:  str
    final_answer: str


# ── Helper: build a standard LLM + ToolNode agent sub-graph ──────────────────

def build_agent(system_prompt: str, tools: list) -> object:
    """Build a compiled ToolNode agent with a given system prompt and tools."""

    bound_llm = llm.bind_tools(tools) if tools else llm
    system    = SystemMessage(content=system_prompt)

    def llm_node(state: State) -> dict:
        response = bound_llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(State)
    builder.add_node("llm",   llm_node)

    if tools:
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "llm")
        builder.add_conditional_edges("llm", tools_condition)
        builder.add_edge("tools", "llm")
    else:
        builder.add_edge(START, "llm")
        builder.add_edge("llm", END)

    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# JIRA SPECIALIST
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def create_jira_ticket(title: str, description: str, priority: str = "medium",
                       assignee: Optional[str] = None) -> str:
    """Create a Jira ticket."""
    tid = f"PROJ-{abs(hash(title)) % 9000 + 1000}"
    return f"[JIRA] {tid} created: '{title}' [{priority}]" + (f" → @{assignee}" if assignee else "")


@tool
def search_jira(query: str) -> str:
    """Search Jira tickets by keyword."""
    return f"[JIRA] Found for '{query}': PROJ-101 (Login bug), PROJ-202 (Deploy failure)"


@tool
def get_sprint_status(sprint_name: str) -> str:
    """Get the status of a Jira sprint."""
    return f"[JIRA] Sprint '{sprint_name}': 12 stories, 8 done, 3 in progress, 1 blocked"


jira_agent = build_agent(
    system_prompt=(
        "You are a Jira specialist. Help with tickets, sprints, epics, and project tracking. "
        "Always use tools to take action. Be concise and practical."
    ),
    tools=[create_jira_ticket, search_jira, get_sprint_status],
)


# ═══════════════════════════════════════════════════════════════════════════════
# AWS SPECIALIST
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_ec2_instances(region: str = "us-east-1") -> str:
    """List running EC2 instances in a region."""
    return f"[AWS] {region}: i-0abc123 (t3.medium, web), i-0def456 (t3.large, api)"


@tool
def get_aws_cost(month: str) -> str:
    """Get AWS cost breakdown for a given month (YYYY-MM)."""
    return f"[AWS] {month}: EC2 $312, RDS $98, S3 $15, Lambda $4, Total $429"


@tool
def describe_security_group(group_id: str) -> str:
    """Describe an AWS security group's inbound and outbound rules."""
    return f"[AWS] {group_id}: inbound 443 (0.0.0.0/0), 22 (10.0.0.0/8); outbound all"


aws_agent = build_agent(
    system_prompt=(
        "You are an AWS solutions architect. Help with EC2, S3, Lambda, RDS, IAM, costs, "
        "and best practices. Use tools when available. Be concise."
    ),
    tools=[get_ec2_instances, get_aws_cost, describe_security_group],
)


# ═══════════════════════════════════════════════════════════════════════════════
# KUBERNETES SPECIALIST
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_pod_status(namespace: str = "default") -> str:
    """List pod statuses in a Kubernetes namespace."""
    return (
        f"[K8S] {namespace}: payments-abc (Running), auth-xyz (CrashLoopBackOff), "
        "api-123 (Running)"
    )


@tool
def describe_pod(pod_name: str, namespace: str = "default") -> str:
    """Get detailed information about a specific Kubernetes pod."""
    return (
        f"[K8S] {pod_name} in {namespace}: "
        "OOMKilled (exit 137), memory limit 256Mi, last restart 3m ago"
    )


@tool
def scale_deployment(deployment: str, replicas: int, namespace: str = "default") -> str:
    """Scale a Kubernetes deployment to the specified number of replicas."""
    return f"[K8S] {deployment} in {namespace} scaled to {replicas} replicas. ✓"


kubernetes_agent = build_agent(
    system_prompt=(
        "You are a Kubernetes expert. Help with pods, deployments, services, Helm, "
        "namespaces, and kubectl. Use tools when available. Be concise."
    ),
    tools=[get_pod_status, describe_pod, scale_deployment],
)


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL SPECIALIST (no tools — pure LLM)
# ═══════════════════════════════════════════════════════════════════════════════

general_agent = build_agent(
    system_prompt=(
        "You are a helpful DevOps assistant. Answer general technical questions "
        "clearly and concisely."
    ),
    tools=[],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test each specialist independently
# ═══════════════════════════════════════════════════════════════════════════════

def test_specialist(name: str, agent, question: str) -> None:
    print(f"\n{'─'*60}")
    print(f"[{name}] {question}")
    result = agent.invoke({
        "messages":     [HumanMessage(content=question)],
        "next_agent":   "",
        "final_answer": "",
    })
    print(f"Answer: {result['messages'][-1].content}")


test_specialist("JIRA", jira_agent,
    "Create a critical Jira ticket: prod API returning 503 since deploy.")

test_specialist("AWS", aws_agent,
    "What are our running EC2 instances in us-east-1?")

test_specialist("KUBERNETES", kubernetes_agent,
    "The auth pod is in CrashLoopBackOff — what's wrong?")

test_specialist("GENERAL", general_agent,
    "What is the difference between blue-green and canary deployments?")
