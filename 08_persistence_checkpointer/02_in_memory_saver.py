"""
Concept: InMemorySaver checkpointer — persistent state across invocations.

Without a checkpointer, every app.invoke() starts with a blank state.
The graph has no memory of previous calls.

With InMemorySaver, LangGraph snapshots the full state after every node
and stores it in a dict keyed by thread_id + checkpoint_id.
On the next call with the same thread_id, the graph resumes from the
last snapshot — all prior messages are already in state.

This is how a chatbot "remembers" the conversation without the caller
managing a history list.

Run:
    python 08_persistence_checkpointer/02_in_memory_saver.py
"""

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ── State & graph ─────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)

# ── Compile WITH a checkpointer ───────────────────────────────────────────────
# MemorySaver stores snapshots in a Python dict — fine for dev/testing.
# Swap for AsyncPostgresSaver in production (section 04).

checkpointer = MemorySaver()
app = graph = builder.compile(checkpointer=checkpointer)

# ── thread_id — the key that identifies one conversation ─────────────────────
# All invocations with the same thread_id share a single history.
# Different thread_ids are completely independent.

config = {"configurable": {"thread_id": "session-001"}}


# ── Turn 1 ────────────────────────────────────────────────────────────────────
print("=== Turn 1 ===")
result = app.invoke(
    {"messages": [HumanMessage(content="My name is Alex. I work on the auth team.")]},
    config=config,
)
print("AI:", result["messages"][-1].content)


# ── Turn 2 — the graph remembers turn 1 without the caller managing history ──
print("\n=== Turn 2 ===")
result = app.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config,
)
print("AI:", result["messages"][-1].content)


# ── Turn 3 ────────────────────────────────────────────────────────────────────
print("\n=== Turn 3 ===")
result = app.invoke(
    {"messages": [HumanMessage(content="Which team do I work on?")]},
    config=config,
)
print("AI:", result["messages"][-1].content)


# ── Inspect the checkpoint ────────────────────────────────────────────────────
print("\n=== Checkpoint state ===")
snapshot = app.get_state(config)
print(f"Messages in checkpoint : {len(snapshot.values['messages'])}")
for msg in snapshot.values["messages"]:
    print(f"  [{msg.type:8}] {msg.content[:70]}")


# ── Without a checkpointer — no memory ───────────────────────────────────────
print("\n=== Without checkpointer (stateless) ===")
stateless_app = builder.compile()   # no checkpointer

r1 = stateless_app.invoke({"messages": [HumanMessage(content="My name is Sam.")]})
print("Turn 1 AI:", r1["messages"][-1].content[:60])

r2 = stateless_app.invoke({"messages": [HumanMessage(content="What's my name?")]})
print("Turn 2 AI:", r2["messages"][-1].content[:60])
# Model won't know the name — each call is independent
