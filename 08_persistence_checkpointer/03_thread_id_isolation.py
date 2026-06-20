"""
Concept: thread_id — isolated conversations within the same graph.

The same compiled app can handle thousands of concurrent conversations.
Each gets its own thread_id. The checkpointer stores state per thread_id
so histories never bleed across users or sessions.

This is how a production chatbot serves multiple users with one deployment:
  - user_A's history  →  thread_id="user_A"
  - user_B's history  →  thread_id="user_B"
  - support_session   →  thread_id="ticket-4521"

Run:
    python 08_persistence_checkpointer/03_thread_id_isolation.py
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


# ── Graph (shared by all conversations) ──────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)

app = builder.compile(checkpointer=MemorySaver())


# ── Helper ────────────────────────────────────────────────────────────────────

def chat(thread_id: str, user_message: str) -> str:
    """Send a message in a specific thread and return the AI reply."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    return result["messages"][-1].content


# ── Conversation A — Alice ────────────────────────────────────────────────────
print("=== Thread: alice ===")
print("A1:", chat("alice", "Hi, my name is Alice and I'm debugging a Kubernetes issue."))
print("A2:", chat("alice", "What's my name?"))
print("A3:", chat("alice", "What am I working on?"))


# ── Conversation B — Bob (completely independent) ─────────────────────────────
print("\n=== Thread: bob ===")
print("B1:", chat("bob", "Hey, I'm Bob. I work on the AWS billing team."))
print("B2:", chat("bob", "Who am I?"))


# ── Cross-thread isolation check ──────────────────────────────────────────────
# Alice's thread should have no knowledge of Bob's messages and vice versa.
print("\n=== Isolation check ===")
alice_reply = chat("alice", "What team do I work on?")
bob_reply   = chat("bob",   "What Kubernetes issue am I debugging?")

print(f"Alice asked about her team   : {alice_reply[:100]}")
print(f"Bob asked about k8s (knows nothing): {bob_reply[:100]}")


# ── Inspect each thread's checkpoint ─────────────────────────────────────────
print("\n=== Checkpoint sizes ===")
for thread_id in ["alice", "bob"]:
    config  = {"configurable": {"thread_id": thread_id}}
    snapshot = app.get_state(config)
    msgs     = snapshot.values.get("messages", [])
    print(f"  thread={thread_id!r:8}  messages={len(msgs)}")
    for m in msgs:
        print(f"    [{m.type:8}] {m.content[:60]}")


# ── Third thread — a support ticket session ───────────────────────────────────
print("\n=== Thread: ticket-9001 ===")
print(chat("ticket-9001", "This thread is for tracking the prod outage on 2026-06-20."))
print(chat("ticket-9001", "Root cause was a misconfigured Nginx ingress after the deploy."))
print(chat("ticket-9001", "What was the root cause of the outage?"))
