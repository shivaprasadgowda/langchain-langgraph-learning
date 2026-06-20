"""
Concept: tools_condition — the routing function for ToolNode agents.

tools_condition is a pre-built routing function designed to work with ToolNode:

    if last message has tool_calls → return "tools"  (route to ToolNode)
    else                           → return END       (model is done)

It is passed directly to add_conditional_edges and replaces the
hand-written route_by_intent or custom routing functions.

Graph shape with tools_condition:

    START
      │
      ▼
    llm_node  ────────────────────► END
      ▲            (no tool calls)
      │
      │        (tool_calls present)
      ▼
    tool_node  (ToolNode)

This creates a cycle: llm → tools → llm → tools → ... → END

Run:
    python 10_toolnode_agents/02_tools_condition.py
"""

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_docs(query: str) -> str:
    """Search the internal documentation for a given query."""
    return f"[DOCS] Results for '{query}': LangGraph uses StateGraph with nodes and edges."


@tool
def get_ticket_status(ticket_id: str) -> str:
    """Get the current status of a Jira ticket by its ID."""
    statuses = {"PROJ-101": "In Progress", "PROJ-202": "Done", "PROJ-303": "Open"}
    return f"[JIRA] {ticket_id}: {statuses.get(ticket_id, 'Not found')}"


tools = [search_docs, get_ticket_status]


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── LLM node ──────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


# ── Build graph with tools_condition ─────────────────────────────────────────

builder = StateGraph(State)

builder.add_node("llm",   llm_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "llm")

# tools_condition inspects the last message:
#   tool_calls present → "tools"
#   no tool_calls      → END
builder.add_conditional_edges("llm", tools_condition)

# After tools run, always go back to llm (creates the agent loop cycle)
builder.add_edge("tools", "llm")

app = builder.compile()


# ── Demonstrate the routing logic directly ────────────────────────────────────

from langchain_core.messages import AIMessage

print("=== tools_condition routing ===")

# Case 1: AIMessage WITH tool_calls → should return "tools"
state_with_tools = {"messages": [AIMessage(
    content="",
    tool_calls=[{"name": "search_docs", "args": {"query": "LangGraph"}, "id": "c1", "type": "tool_call"}]
)]}
print(f"  With tool_calls    → {tools_condition(state_with_tools)!r}")

# Case 2: AIMessage WITHOUT tool_calls → should return END
state_without_tools = {"messages": [AIMessage(content="LangGraph is a graph-based agent framework.")]}
print(f"  Without tool_calls → {tools_condition(state_without_tools)!r}  (END)")


# ── Run the full agent ────────────────────────────────────────────────────────

print("\n=== Agent runs ===")

queries = [
    "What is LangGraph? Search the docs.",
    "What's the status of ticket PROJ-101?",
    "What is 2 + 2?",   # no tool needed — model answers directly
]

for q in queries:
    print(f"\nUser: {q}")
    result = app.invoke({"messages": [HumanMessage(content=q)]})
    print(f"Answer: {result['messages'][-1].content}")
    steps = len([m for m in result["messages"] if hasattr(m, "tool_call_id")])
    print(f"Tool calls executed: {steps}")
