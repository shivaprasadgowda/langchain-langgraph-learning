"""
Concept: A minimal LangGraph StateGraph with one LLM node.

Graph shape:
    START → llm_node → END

State holds a list of messages.
The llm_node calls the model and returns the reply.
add_messages ensures replies are appended, not overwritten.

Run:
    python 06_langgraph_basics/02_first_graph.py
"""

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


# ── 1. Define State ───────────────────────────────────────────────────────────
# State is a TypedDict — a typed dict that every node reads from and writes to.
# Annotated[list, add_messages] tells LangGraph to append new messages
# rather than replace the list when a node returns {"messages": [...]}.

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. Define node(s) ─────────────────────────────────────────────────────────
# A node is a plain function: State → dict (partial state update).
# LangGraph merges the returned dict into the current state.

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def llm_node(state: State) -> dict:
    """Call the LLM with the current message history and return the reply."""
    response = llm.invoke(state["messages"])
    # Return only the fields we are updating — not the full state
    return {"messages": [response]}


# ── 3. Build the graph ────────────────────────────────────────────────────────

graph_builder = StateGraph(State)

graph_builder.add_node("llm", llm_node)

# Edges: START → llm → END
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)

# ── 4. Compile ────────────────────────────────────────────────────────────────
# compile() validates the graph and returns an executable Runnable.
# Without a checkpointer, state is not persisted between invocations.

app = graph_builder.compile()


# ── 5. Invoke ─────────────────────────────────────────────────────────────────
# Pass the initial state. add_messages will merge these into an empty list.

initial_state = {"messages": [HumanMessage(content="What is LangGraph in one sentence?")]}
result = app.invoke(initial_state)

print("=== Final state ===")
for msg in result["messages"]:
    print(f"  [{msg.type}] {msg.content}")


# ── 6. Stream — receive output from each node as it runs ──────────────────────
print("\n=== Streamed events ===")
for event in app.stream({"messages": [HumanMessage(content="Name one advantage of LangGraph.")]}):
    for node_name, state_update in event.items():
        print(f"  node={node_name!r}")
        for msg in state_update.get("messages", []):
            print(f"    [{msg.type}] {msg.content[:80]}")
