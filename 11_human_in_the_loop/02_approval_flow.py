"""
Concept: Pause → human input → resume using interrupt().

The interrupt() function (Method B) is more flexible than interrupt_before:
  - It pauses from inside any node
  - It can pass a value TO the human (e.g. "please review: {details}")
  - The human's response is returned as the return value of interrupt()
  - Resume via app.invoke(Command(resume=human_reply), config=config)

Flow:
  user message
       │
       ▼
  plan_node       ← decides what action to take, stores in state
       │
       ▼
  approval_node   ← calls interrupt("approve?") → pauses here
       │ (human reviews and calls app.invoke(Command(resume="yes")))
       ▼
  execute_node    ← runs the action only if approved
       │
       ▼
      END

Run:
    python 11_human_in_the_loop/02_approval_flow.py
"""

from dotenv import load_dotenv
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    planned_action: str    # what the agent wants to do
    approved:       bool   # set by approval_node after human responds
    result:         str    # set by execute_node


# ── Planner — decides what action to take ─────────────────────────────────────

class ActionPlan(BaseModel):
    """A planned action the agent proposes to take."""
    action:      str = Field(description="Short description of the action to take.")
    reason:      str = Field(description="Why this action is needed.")
    risk_level:  str = Field(description="'low', 'medium', or 'high' risk.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_plan_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a DevOps assistant. Plan a concrete action to fulfil the user's request."),
        ("human",  "{request}"),
    ])
    | llm.with_structured_output(ActionPlan)
)

def plan_node(state: State) -> dict:
    """Analyse the request and propose an action."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"), ""
    )
    plan: ActionPlan = _plan_chain.invoke({"request": last_human})
    planned = (
        f"Action    : {plan.action}\n"
        f"Reason    : {plan.reason}\n"
        f"Risk level: {plan.risk_level}"
    )
    print(f"  [plan_node] proposed:\n{planned}")
    return {"planned_action": planned}


# ── Approval node — pauses for human review ───────────────────────────────────

def approval_node(state: State) -> dict:
    """Show the plan to the human and wait for approval."""
    prompt = (
        f"The agent wants to take the following action:\n\n"
        f"{state['planned_action']}\n\n"
        f"Type 'yes' to approve or 'no' to reject."
    )
    # interrupt() suspends the graph here and returns the human's response
    # when app.invoke(Command(resume=...), config=config) is called.
    human_response: str = interrupt(prompt)

    approved = human_response.strip().lower() in ("yes", "y", "approve", "ok")
    print(f"  [approval_node] human said: {human_response!r} → approved={approved}")
    return {"approved": approved}


# ── Execute node — runs only if approved ──────────────────────────────────────

def execute_node(state: State) -> dict:
    """Execute the planned action (or report rejection)."""
    if state["approved"]:
        result = f"[EXECUTED] {state['planned_action'].splitlines()[0]}"
        print(f"  [execute_node] {result}")
    else:
        result = "[REJECTED] Action was not approved by human."
        print(f"  [execute_node] {result}")
    return {"result": result}


# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(State)
builder.add_node("plan",     plan_node)
builder.add_node("approval", approval_node)
builder.add_node("execute",  execute_node)

builder.add_edge(START,      "plan")
builder.add_edge("plan",     "approval")
builder.add_edge("approval", "execute")
builder.add_edge("execute",  END)

app = builder.compile(checkpointer=MemorySaver())


# ── Scenario 1: Human approves ────────────────────────────────────────────────

print("=" * 60)
print("Scenario 1: Human APPROVES")
print("=" * 60)

config1 = {"configurable": {"thread_id": "flow-001"}}
initial = {
    "messages":       [HumanMessage(content="Restart the payment-service pod in production.")],
    "planned_action": "",
    "approved":       False,
    "result":         "",
}

# First invoke — pauses at approval_node
app.invoke(initial, config=config1)

# Inspect what the human sees
snapshot = app.get_state(config1)
print("\n  Graph paused. Pending at:", snapshot.next)

# Resume with approval
print("\n  Human types: 'yes'")
final1 = app.invoke(Command(resume="yes"), config=config1)
print(f"\n  Result: {final1['result']}")


# ── Scenario 2: Human rejects ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Scenario 2: Human REJECTS")
print("=" * 60)

config2 = {"configurable": {"thread_id": "flow-002"}}
app.invoke(
    {**initial, "messages": [HumanMessage(content="Delete all logs older than 7 days from prod.")]},
    config=config2,
)

print("\n  Human types: 'no'")
final2 = app.invoke(Command(resume="no"), config=config2)
print(f"\n  Result: {final2['result']}")
