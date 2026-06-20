"""
Concept: How interrupt works in LangGraph.

LangGraph provides two ways to pause a graph mid-execution:

  Method A — interrupt_before (compile-time):
    Pause before a named node runs.
    graph.compile(checkpointer=..., interrupt_before=["node_name"])
    Resume by calling app.invoke(None, config=config)

  Method B — interrupt() function (node-time, modern preferred):
    Pause from inside a node and wait for a value.
    from langgraph.types import interrupt
    def my_node(state): value = interrupt("Prompt for human")
    Resume by calling app.invoke(Command(resume=value), config=config)

This file demonstrates Method A (interrupt_before) because it is the
simplest way to grasp the pause/resume lifecycle.

Key insight: a checkpointer is REQUIRED for interrupt to work.
Without a checkpointer there is nowhere to save the paused state.

Run:
    python 11_human_in_the_loop/01_interrupt_concept.py
"""

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ── Simple two-node graph ─────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_log: list[str]


def node_a(state: State) -> dict:
    print("  [node_a] running")
    return {"step_log": state.get("step_log", []) + ["node_a completed"]}


def node_b(state: State) -> dict:
    print("  [node_b] running  ← this is the action we want approval for")
    return {"step_log": state.get("step_log", []) + ["node_b completed"]}


builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START,    "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

# A checkpointer is mandatory for interrupt to work
checkpointer = MemorySaver()

# interrupt_before=["node_b"] pauses after node_a completes
# but before node_b starts
app = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["node_b"],
)

config = {"configurable": {"thread_id": "demo-interrupt-001"}}


# ── First invoke — graph pauses before node_b ─────────────────────────────────

print("=== First invoke (will pause before node_b) ===")
result = app.invoke(
    {
        "messages":  [HumanMessage(content="trigger the graph")],
        "step_log":  [],
    },
    config=config,
)
print("invoke() returned (graph is paused, not finished)")
print("step_log so far:", result.get("step_log", []))


# ── Inspect the paused state ──────────────────────────────────────────────────

print("\n=== Paused state ===")
snapshot = app.get_state(config)
print("values['step_log'] :", snapshot.values.get("step_log"))
print("next nodes to run  :", snapshot.next)   # ('node_b',)
print("graph is done?     :", snapshot.next == ())


# ── Resume — pass None to continue from where we left off ───────────────────
# Passing None means "no new input — just resume"

print("\n=== Resuming (human approved) ===")
final = app.invoke(None, config=config)   # resume
print("step_log after resume:", final.get("step_log", []))


# ── State after completion ────────────────────────────────────────────────────

print("\n=== Final snapshot ===")
done = app.get_state(config)
print("next nodes :", done.next)          # () — graph finished
print("step_log   :", done.values.get("step_log"))


# ── Reject: update state before resuming ─────────────────────────────────────
print("\n=== Reject example (new thread) ===")

config2 = {"configurable": {"thread_id": "demo-interrupt-002"}}

app.invoke({"messages": [HumanMessage(content="trigger")], "step_log": []}, config=config2)
snapshot2 = app.get_state(config2)
print("Paused. Next:", snapshot2.next)

# Human rejects — update state to mark rejection instead of resuming
app.update_state(
    config2,
    {"step_log": snapshot2.values.get("step_log", []) + ["node_b REJECTED by human"]},
    as_node="node_b",  # treat the update as if node_b already ran
)

done2 = app.get_state(config2)
print("After rejection update, step_log:", done2.values.get("step_log"))
print("next:", done2.next)   # () — graph considers node_b done
