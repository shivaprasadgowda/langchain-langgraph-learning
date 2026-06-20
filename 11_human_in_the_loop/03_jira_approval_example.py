"""
Concept: Realistic human-in-the-loop — approve before Jira ticket creation.

This example combines the ToolNode agent pattern (section 10) with
human-in-the-loop approval using interrupt_before.

The agent:
  1. Classifies the request — Jira-related or not?
  2. If Jira: extracts the ticket details and shows them for approval.
  3. Human approves → ticket is created.
  4. Human rejects → agent reports the rejection, no ticket created.
  5. Non-Jira requests bypass approval entirely.

Graph shape:
    START
      │
      ▼
   llm_node ──(no tool_calls)──► END
      │
      │ (tool_calls present)
      ▼
  [PAUSE — interrupt_before]
      │
      │ (human approves: invoke(None, config))
      ▼
   tools_node (ToolNode — actually creates the ticket)
      │
      ▼
   llm_node  (model summarises result)
      │
      ▼
      END

Run:
    python 11_human_in_the_loop/03_jira_approval_example.py
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
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def create_jira_ticket(
    title:       str,
    description: str,
    priority:    str = "medium",
    assignee:    Optional[str] = None,
) -> str:
    """
    Create a Jira ticket in the project board.

    Args:
        title:       Short ticket title (max 10 words).
        description: Full description of the issue or task.
        priority:    One of 'low', 'medium', 'high', 'critical'.
        assignee:    Jira username to assign, if mentioned.
    """
    ticket_id = f"PROJ-{abs(hash(title)) % 9000 + 1000}"
    assignee_str = f" → @{assignee}" if assignee else ""
    return (
        f"[JIRA] Ticket created: {ticket_id}\n"
        f"  Title    : {title}\n"
        f"  Priority : {priority}{assignee_str}\n"
        f"  Status   : Open"
    )


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 19°C and overcast."


tools    = [create_jira_ticket, get_weather]
tool_map = {t.name: t for t in tools}


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Nodes ─────────────────────────────────────────────────────────────────────

SYSTEM = SystemMessage(content=(
    "You are a DevOps assistant. You can create Jira tickets and check weather. "
    "When creating a Jira ticket, extract a clear title, description, priority, "
    "and assignee (if mentioned). Be concise in your final response."
))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke([SYSTEM] + state["messages"])]}


# ── Graph with interrupt_before=["tools"] ────────────────────────────────────
# The graph pauses before ToolNode runs whenever the LLM has requested tools.
# This gives the human a chance to review before any tool executes.

builder = StateGraph(State)
builder.add_node("llm",   llm_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")

app = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],   # pause before any tool runs
)


# ── Helper: show pending tool calls ──────────────────────────────────────────

def show_pending_tools(config: dict) -> None:
    snapshot  = app.get_state(config)
    last_msg  = snapshot.values["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print("  Pending tool calls:")
        for tc in last_msg.tool_calls:
            print(f"    Tool    : {tc['name']}")
            for k, v in tc["args"].items():
                print(f"    {k:<12}: {v}")
    print(f"  Next nodes: {snapshot.next}")


# ── Scenario 1: Jira ticket — human approves ─────────────────────────────────

print("=" * 65)
print("Scenario 1: Jira ticket — APPROVED")
print("=" * 65)

c1 = {"configurable": {"thread_id": "jira-001"}}
app.invoke(
    {"messages": [HumanMessage(
        content="Create a critical Jira ticket: prod login page 500 error since deploy. Assign to @sarah."
    )]},
    config=c1,
)

print("\n  [PAUSED] Agent wants to create a Jira ticket. Review:")
show_pending_tools(c1)

print("\n  Human decision: APPROVE")
final1 = app.invoke(None, config=c1)   # None = resume with no changes
print(f"\n  Agent reply: {final1['messages'][-1].content}")


# ── Scenario 2: Jira ticket — human rejects ───────────────────────────────────

print("\n" + "=" * 65)
print("Scenario 2: Jira ticket — REJECTED")
print("=" * 65)

c2 = {"configurable": {"thread_id": "jira-002"}}
app.invoke(
    {"messages": [HumanMessage(
        content="Create a low-priority Jira ticket: update README with new env vars."
    )]},
    config=c2,
)

print("\n  [PAUSED] Agent wants to create a Jira ticket. Review:")
show_pending_tools(c2)

print("\n  Human decision: REJECT — remove the tool call and let model respond directly")

# To reject: update state to remove the tool_calls from the last AIMessage,
# replacing it with an AIMessage that has no tool_calls so tools_condition
# will route to END on next invoke.
from langchain_core.messages import AIMessage

snapshot2 = app.get_state(c2)
last_ai   = snapshot2.values["messages"][-1]

# Replace the tool-calling AIMessage with a plain one signalling rejection
app.update_state(
    c2,
    {"messages": [AIMessage(
        content="I was going to create a Jira ticket but the action was rejected by a human reviewer. "
                "No ticket has been created.",
        id=last_ai.id,   # same id so add_messages deduplicates (replaces it)
    )]},
)

final2 = app.invoke(None, config=c2)
print(f"\n  Agent reply: {final2['messages'][-1].content}")


# ── Scenario 3: Non-Jira request — bypasses approval ─────────────────────────

print("\n" + "=" * 65)
print("Scenario 3: Weather request — no approval needed")
print("=" * 65)

c3 = {"configurable": {"thread_id": "weather-001"}}
result3 = app.invoke(
    {"messages": [HumanMessage(content="What's the weather in Amsterdam?")]},
    config=c3,
)

snapshot3 = app.get_state(c3)
if snapshot3.next:
    print("  [PAUSED] Approval needed for weather tool.")
    final3 = app.invoke(None, config=c3)
    print(f"  Agent reply: {final3['messages'][-1].content}")
else:
    print(f"  Agent reply: {result3['messages'][-1].content}")
    print("  (graph completed — no pause triggered)")
