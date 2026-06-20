"""
Concept: State, Node, and Edge — each primitive in isolation.

This file examines each LangGraph building block on its own so the
mechanics are clear before combining them into larger graphs.

Run:
    python 06_langgraph_basics/03_state_node_edge.py
"""

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE 1: STATE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PRIMITIVE 1: STATE")
print("=" * 60)

# State is a TypedDict. Every node receives it and returns a partial update.
# Fields without a reducer use last-write-wins (new value replaces old).
# Fields with a reducer use the reducer to merge (e.g. add_messages appends).

class AgentState(TypedDict):
    messages:  Annotated[list[BaseMessage], add_messages]  # append-on-update
    intent:    str                                          # last-write-wins
    step_count: int                                         # last-write-wins

# Demonstrate reducer behaviour directly
from langgraph.graph.message import add_messages as _add

existing = [HumanMessage(content="Hello")]
incoming = [HumanMessage(content="How are you?")]
merged   = _add(existing, incoming)

print("add_messages reducer:")
print("  existing :", [m.content for m in existing])
print("  incoming :", [m.content for m in incoming])
print("  merged   :", [m.content for m in merged])   # both messages present


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE 2: NODE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PRIMITIVE 2: NODE")
print("=" * 60)

# A node is: (state: State) -> dict
# - Receives the full state.
# - Returns ONLY the fields it wants to update (partial update).
# - LangGraph merges the returned dict into the state.

def classify_node(state: AgentState) -> dict:
    """Classify the intent of the last human message."""
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"), None
    )
    text = last_human.content.lower() if last_human else ""

    if "weather" in text:
        intent = "weather"
    elif "jira" in text or "ticket" in text:
        intent = "jira"
    else:
        intent = "general"

    print(f"  [classify_node] intent={intent!r}")
    # Only return fields we changed — messages is untouched
    return {"intent": intent, "step_count": state.get("step_count", 0) + 1}


def llm_node(state: AgentState) -> dict:
    """Call the LLM and append its reply to messages."""
    system = SystemMessage(content=f"You are a helpful assistant. User intent: {state.get('intent', 'general')}.")
    response = llm.invoke([system] + state["messages"])
    print(f"  [llm_node] reply: {response.content[:60]}...")
    return {"messages": [response], "step_count": state.get("step_count", 0) + 1}


# Demonstrate calling a node function directly (no graph needed)
sample_state: AgentState = {
    "messages": [HumanMessage(content="Create a Jira ticket for the login bug.")],
    "intent": "",
    "step_count": 0,
}
update = classify_node(sample_state)
print("Node returned (partial update):", update)


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE 3: EDGES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PRIMITIVE 3: EDGES")
print("=" * 60)

# Normal edge: always routes to the same next node.
# graph.add_edge("node_a", "node_b")

# Conditional edge: calls a function on state, returns a node name.
# The returned string must match a registered node name or END.
# graph.add_conditional_edges("node_a", routing_fn)

def route_by_intent(state: AgentState) -> str:
    """Routing function: returns the next node name based on intent."""
    intent = state.get("intent", "general")
    destination = {
        "weather": "weather_node",
        "jira":    "jira_node",
    }.get(intent, "llm_node")   # default to llm_node for general
    print(f"  [router] intent={intent!r} → {destination!r}")
    return destination


# ═══════════════════════════════════════════════════════════════════════════════
# PUTTING IT TOGETHER: two-node graph with conditional routing
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FULL EXAMPLE: classify → conditional route → llm")
print("=" * 60)

def weather_node(state: AgentState) -> dict:
    return {"messages": [HumanMessage(content="[stub] It is sunny today.")]}

def jira_node(state: AgentState) -> dict:
    return {"messages": [HumanMessage(content="[stub] Jira ticket PROJ-42 created.")]}

builder = StateGraph(AgentState)
builder.add_node("classify", classify_node)
builder.add_node("llm_node", llm_node)
builder.add_node("weather_node", weather_node)
builder.add_node("jira_node", jira_node)

# START → classify (always)
builder.add_edge(START, "classify")

# classify → one of three nodes (conditional)
builder.add_conditional_edges(
    "classify",
    route_by_intent,
    # Explicit mapping: return value → node name (optional but documents intent)
    {
        "llm_node":     "llm_node",
        "weather_node": "weather_node",
        "jira_node":    "jira_node",
    },
)

# All specialist nodes → END
builder.add_edge("llm_node",     END)
builder.add_edge("weather_node", END)
builder.add_edge("jira_node",    END)

app = builder.compile()

for question in [
    "What is a REST API?",
    "Create a Jira ticket for the deploy failure.",
]:
    print(f"\nUser: {question}")
    result = app.invoke({
        "messages": [HumanMessage(content=question)],
        "intent": "",
        "step_count": 0,
    })
    final_msg = result["messages"][-1]
    print(f"Answer ({result['intent']}): {final_msg.content[:80]}")
    print(f"Steps taken: {result['step_count']}")
