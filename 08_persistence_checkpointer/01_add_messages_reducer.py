"""
Concept: The add_messages reducer — how chat history accumulates.

Without a reducer, every node return value replaces the existing field:
    state["messages"] = node_return["messages"]   ← overwrites

With add_messages, LangGraph calls the reducer instead:
    state["messages"] = add_messages(existing, node_return["messages"])

add_messages appends new messages and deduplicates by message id.
Deduplication matters because checkpointers replay state on resume —
the same message must never appear twice in history.

This file explores the reducer directly (no graph needed).

Run:
    python 08_persistence_checkpointer/01_add_messages_reducer.py
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages


# ── 1. Basic append behaviour ─────────────────────────────────────────────────

existing = [
    SystemMessage(content="You are helpful."),
    HumanMessage(content="Hello!"),
]
incoming = [AIMessage(content="Hi! How can I help?")]

merged = add_messages(existing, incoming)

print("=== Basic append ===")
for m in merged:
    print(f"  [{m.type}] {m.content}")
# All three messages present


# ── 2. Deduplication by id ────────────────────────────────────────────────────
# If a message with the same id appears in both lists, the incoming one wins
# (it is treated as an update to the existing message).

msg_a = HumanMessage(content="original", id="msg-1")
msg_b = HumanMessage(content="updated",  id="msg-1")   # same id, new content

deduped = add_messages([msg_a], [msg_b])

print("\n=== Deduplication (same id) ===")
for m in deduped:
    print(f"  id={m.id!r}  content={m.content!r}")
# Only one message — the incoming (updated) version


# ── 3. What last-write-wins looks like (without a reducer) ────────────────────
# For comparison: a plain list field gets replaced entirely.

print("\n=== Last-write-wins (no reducer) ===")

state_no_reducer = {"plain_list": [HumanMessage(content="turn 1")]}
node_return      = {"plain_list": [AIMessage(content="turn 2")]}

# LangGraph without a reducer simply does:
state_no_reducer["plain_list"] = node_return["plain_list"]

print("  Result:", [m.content for m in state_no_reducer["plain_list"]])
# ['turn 2'] — turn 1 is lost


# ── 4. Simulate two nodes writing to the same messages field ──────────────────

print("\n=== Two node writes accumulated ===")
history = []

# Node A runs
history = add_messages(history, [HumanMessage(content="What is LangGraph?")])
# LLM node runs
history = add_messages(history, [AIMessage(content="LangGraph is a graph-based agent framework.")])
# User sends another message
history = add_messages(history, [HumanMessage(content="How does state work?")])
# LLM node runs again
history = add_messages(history, [AIMessage(content="State is a TypedDict shared by all nodes.")])

for i, m in enumerate(history):
    print(f"  [{i}] {m.type:8} : {m.content}")


# ── 5. Remove a message using RemoveMessage ───────────────────────────────────
# add_messages also supports deletion via RemoveMessage(id=...).

from langchain_core.messages import RemoveMessage

msg1 = HumanMessage(content="keep me",   id="keep")
msg2 = HumanMessage(content="delete me", id="del")

after_add    = add_messages([], [msg1, msg2])
after_remove = add_messages(after_add, [RemoveMessage(id="del")])

print("\n=== RemoveMessage ===")
print("  Before:", [m.content for m in after_add])
print("  After :", [m.content for m in after_remove])
