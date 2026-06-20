"""
Concept: Full multi-agent system — supervisor + specialist sub-graphs.

The parent graph:
  1. Runs the supervisor node to classify intent and set next_agent.
  2. Routes to the appropriate specialist sub-graph via a conditional edge.
  3. The specialist sub-graph runs its own agent loop (LLM + tools).
  4. The specialist's final state is merged back into the parent state.
  5. A response_node formats the final answer.

In LangGraph, a compiled sub-graph can be added as a node with:
    parent.add_node("jira_agent", jira_compiled_app)

The sub-graph receives the full parent state and returns a partial update.
Both graphs must share compatible state schemas (same field names/types).

Full graph shape:
    START
      │
      ▼
  supervisor_node  (classifies → sets next_agent)
      │
      ▼ (conditional edge)
  ┌───┴─────────────────────┐
  │          │              │              │
  ▼          ▼              ▼              ▼
jira_agent  aws_agent  kubernetes_agent  general_agent
  │          │              │              │
  └──────────┴──────────────┴──────────────┘
                            │
                     response_node  (summarise)
                            │
                           END

Run:
    python 12_multi_agent_systems/03_full_multi_agent_graph.py
"""

from dotenv import load_dotenv
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ═══════════════════════════════════════════════════════════════════════════════

class State(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    next_agent:   str
    final_answer: str


# ═══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR
# ═══════════════════════════════════════════════════════════════════════════════

class SupervisorDecision(BaseModel):
    next: Literal["jira_agent", "aws_agent", "kubernetes_agent", "general_agent"]
    reasoning: str


_supervisor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a supervisor routing requests to specialist agents. Choose precisely."),
        ("human",  "{request}"),
    ])
    | llm.with_structured_output(SupervisorDecision)
)

def supervisor_node(state: State) -> dict:
    last_human = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"), ""
    )
    decision: SupervisorDecision = _supervisor_chain.invoke({"request": last_human})
    print(f"  [supervisor] → {decision.next!r}  ({decision.reasoning})")
    return {"next_agent": decision.next}

def route_to_agent(state: State) -> str:
    return state["next_agent"]


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALIST TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def create_jira_ticket(title: str, priority: str = "medium",
                       assignee: Optional[str] = None) -> str:
    """Create a Jira ticket with a title and priority."""
    tid = f"PROJ-{abs(hash(title)) % 9000 + 1000}"
    return f"[JIRA] {tid}: '{title}' [{priority}]" + (f" @{assignee}" if assignee else "")

@tool
def search_jira(query: str) -> str:
    """Search Jira for tickets matching a keyword."""
    return f"[JIRA] '{query}': PROJ-101 (API errors), PROJ-202 (Deploy failure)"

@tool
def get_aws_cost(month: str) -> str:
    """Get AWS cost summary for YYYY-MM."""
    return f"[AWS] {month}: EC2 $312, RDS $98, S3 $15, Total $425"

@tool
def get_ec2_instances(region: str = "us-east-1") -> str:
    """List EC2 instances in a region."""
    return f"[AWS] {region}: i-0abc (t3.medium, web), i-0def (t3.large, api)"

@tool
def get_pod_status(namespace: str = "default") -> str:
    """List pod statuses in a Kubernetes namespace."""
    return f"[K8S] {namespace}: payments (Running), auth (CrashLoopBackOff)"

@tool
def describe_pod(pod_name: str, namespace: str = "default") -> str:
    """Describe a specific Kubernetes pod."""
    return f"[K8S] {pod_name}: OOMKilled (exit 137), memory limit 256Mi"

@tool
def scale_deployment(deployment: str, replicas: int) -> str:
    """Scale a Kubernetes deployment."""
    return f"[K8S] {deployment} scaled to {replicas} replicas ✓"


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALIST SUB-GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_specialist(system_prompt: str, tools: list):
    """Build and compile a ToolNode specialist agent sub-graph."""
    bound_llm = llm.bind_tools(tools) if tools else llm
    system    = SystemMessage(content=system_prompt)

    def llm_node(state: State) -> dict:
        return {"messages": [bound_llm.invoke([system] + state["messages"])]}

    sub = StateGraph(State)
    sub.add_node("llm",   llm_node)
    if tools:
        sub.add_node("tools", ToolNode(tools))
        sub.add_edge(START, "llm")
        sub.add_conditional_edges("llm", tools_condition)
        sub.add_edge("tools", "llm")
    else:
        sub.add_edge(START, "llm")
        sub.add_edge("llm", END)
    return sub.compile()


jira_app = build_specialist(
    "You are a Jira specialist. Create tickets, search issues, manage sprints. Be concise.",
    [create_jira_ticket, search_jira],
)

aws_app = build_specialist(
    "You are an AWS architect. Help with EC2, S3, costs, and best practices. Be concise.",
    [get_aws_cost, get_ec2_instances],
)

kubernetes_app = build_specialist(
    "You are a Kubernetes expert. Help with pods, deployments, and scaling. Be concise.",
    [get_pod_status, describe_pod, scale_deployment],
)

general_app = build_specialist(
    "You are a helpful DevOps assistant. Answer general technical questions concisely.",
    [],
)


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE NODE
# ═══════════════════════════════════════════════════════════════════════════════

def response_node(state: State) -> dict:
    """Extract the specialist's final answer from the last AI message."""
    final = next(
        (m.content for m in reversed(state["messages"]) if m.type == "ai"),
        "No answer produced.",
    )
    return {"final_answer": final}


# ═══════════════════════════════════════════════════════════════════════════════
# PARENT GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

parent = StateGraph(State)

parent.add_node("supervisor",       supervisor_node)
parent.add_node("jira_agent",       jira_app)          # compiled sub-graph as node
parent.add_node("aws_agent",        aws_app)
parent.add_node("kubernetes_agent", kubernetes_app)
parent.add_node("general_agent",    general_app)
parent.add_node("respond",          response_node)

parent.add_edge(START, "supervisor")
parent.add_conditional_edges(
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
    parent.add_edge(agent, "respond")
parent.add_edge("respond", END)

app = parent.compile(checkpointer=MemorySaver())


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def ask(question: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n{'═'*65}")
    print(f"User: {question}")
    result = app.invoke(
        {"messages": [HumanMessage(content=question)], "next_agent": "", "final_answer": ""},
        config=config,
    )
    print(f"Agent [{result['next_agent']}]: {result['final_answer'][:200]}")


ask("Create a critical Jira ticket: prod checkout returning 500 since last deploy.", "t1")
ask("What's our AWS spend for June 2026, and which EC2 instances are running?",     "t2")
ask("The payments pod is OOMKilled — describe it and increase memory.",              "t3")
ask("What is the CAP theorem?",                                                      "t4")
