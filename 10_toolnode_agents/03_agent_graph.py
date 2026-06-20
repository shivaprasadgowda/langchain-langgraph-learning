"""
Concept: Complete ToolNode agent graph with persistence.

This is the production-ready agent pattern:
  - ToolNode handles all tool dispatch
  - tools_condition handles routing
  - MemorySaver provides multi-turn memory
  - System message sets the agent persona
  - Works with parallel tool calls

Graph shape (cyclic):

    START → llm_node ──(tool_calls?)──► tools_node ──┐
               ▲                                       │
               └───────────────────────────────────────┘
               │
            (no tool_calls)
               │
              END

Run:
    python 10_toolnode_agents/03_agent_graph.py
"""

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


@tool
def create_jira_ticket(title: str, priority: str = "medium") -> str:
    """
    Create a Jira ticket.

    Args:
        title:    Short description of the issue.
        priority: One of 'low', 'medium', 'high', 'critical'.
    """
    return f"[JIRA] Created: '{title}' [{priority}] → PROJ-{abs(hash(title)) % 9000 + 1000}"


@tool
def get_aws_cost(month: str) -> str:
    """
    Get AWS cost summary for a given month.

    Args:
        month: Month in YYYY-MM format.
    """
    return f"[AWS] {month} cost: EC2 $312, RDS $98, S3 $15, Total $425"


tools = [add, multiply, create_jira_ticket, get_aws_cost]


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Nodes ─────────────────────────────────────────────────────────────────────

SYSTEM = SystemMessage(content=(
    "You are a DevOps assistant with access to maths, Jira, and AWS tools. "
    "Use tools when they are needed. Be concise."
))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def llm_node(state: State) -> dict:
    # Prepend system message on every call so it is always present
    messages = [SYSTEM] + state["messages"]
    return {"messages": [llm.invoke(messages)]}


# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(State)
builder.add_node("llm",   llm_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)


# ── Single-turn runs ──────────────────────────────────────────────────────────

def ask(question: str, thread_id: str = "demo") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return result["messages"][-1].content


print("=== Single-turn examples ===")
print("\nQ:", "What is 128 divided by 4?   (hint: use multiply by 0.25)")
print("A:", ask("What is 128 multiplied by 0.25?", thread_id="t1"))

print("\nQ:", "Create a critical Jira ticket for the prod login outage.")
print("A:", ask("Create a critical Jira ticket for the prod login outage.", thread_id="t2"))

print("\nQ:", "What did we spend on AWS in May 2026?")
print("A:", ask("What did we spend on AWS in May 2026?", thread_id="t3"))


# ── Multi-turn with memory (same thread_id) ───────────────────────────────────

print("\n=== Multi-turn conversation (thread: support-001) ===")
tid = "support-001"
exchanges = [
    "My name is Jordan. I work on the platform team.",
    "Create a high-priority Jira ticket: Kubernetes pods crashing after latest deploy.",
    "What's my name and what ticket did I just create?",
]

for human_msg in exchanges:
    print(f"\nUser : {human_msg}")
    reply = ask(human_msg, thread_id=tid)
    print(f"Agent: {reply}")


# ── Parallel tool calls ───────────────────────────────────────────────────────

print("\n=== Parallel tool calls ===")
reply = ask(
    "What is 50 multiplied by 3, and also what did we spend on AWS in June 2026? "
    "Answer both in one response.",
    thread_id="t4",
)
print("A:", reply)
