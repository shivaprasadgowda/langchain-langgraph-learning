"""
Concept: graph.stream() — node-level event streaming from LangGraph.

app.invoke() waits for the full graph to finish, then returns final state.
app.stream() yields an event dict after each node completes:

  for event in app.stream(input):
      # event = {node_name: state_update_from_that_node}

Two stream modes:
  "values"  — emits the full accumulated state after each node.
  "updates" — emits only the partial state update each node returned.
               Cheaper to transmit; easier to process for UIs.

Token-level streaming inside a graph:
  Use stream_mode="messages" to receive individual LLM token chunks as
  they are generated inside any node, identified by node name.

Run:
    python 13_streaming/02_graph_stream.py
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

load_dotenv()


# ── Graph setup ───────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary:  str


@tool
def get_server_status(server: str) -> str:
    """Get the status of a named server."""
    statuses = {"web-01": "healthy", "api-01": "degraded", "db-01": "healthy"}
    return f"{server}: {statuses.get(server, 'unknown')}"


tools = [get_server_status]
llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def llm_node(state: State) -> dict:
    system = SystemMessage(content="You are a DevOps assistant with server monitoring tools.")
    return {"messages": [llm.invoke([system] + state["messages"])]}

def summary_node(state: State) -> dict:
    last = state["messages"][-1].content
    return {"summary": f"Done: {last[:60]}..."}

builder = StateGraph(State)
builder.add_node("llm",     llm_node)
builder.add_node("tools",   ToolNode(tools))
builder.add_node("summary", summary_node)

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools",   "llm")
builder.add_edge("llm",     "summary")   # after final LLM answer, summarise
builder.add_edge("summary", END)

# Compile without interrupt so streaming runs to completion
app = builder.compile()

initial = {
    "messages": [HumanMessage(content="What is the status of web-01 and api-01?")],
    "summary":  "",
}


# ── 1. stream_mode="updates" — only what each node changed ───────────────────

print("=" * 60)
print("stream_mode='updates'  (partial state per node)")
print("=" * 60)

for event in app.stream(initial, stream_mode="updates"):
    for node_name, update in event.items():
        print(f"\n[{node_name}]")
        if "messages" in update:
            for msg in update["messages"]:
                label = f"  [{msg.type}]"
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"{label} tool_calls: {[tc['name'] for tc in msg.tool_calls]}")
                elif hasattr(msg, "tool_call_id"):
                    print(f"{label} tool result: {msg.content}")
                else:
                    print(f"{label} {msg.content[:80]}")
        if "summary" in update:
            print(f"  [summary] {update['summary']}")


# ── 2. stream_mode="values" — full accumulated state after each node ──────────

print("\n" + "=" * 60)
print("stream_mode='values'  (full state per node)")
print("=" * 60)

for event in app.stream(initial, stream_mode="values"):
    msg_count   = len(event.get("messages", []))
    last_type   = event["messages"][-1].type if event.get("messages") else "—"
    summary     = event.get("summary", "")
    print(f"  messages={msg_count}  last_type={last_type!r}  summary={summary[:30]!r}")


# ── 3. stream_mode="messages" — token-level chunks from inside nodes ──────────

print("\n" + "=" * 60)
print("stream_mode='messages'  (token chunks + node metadata)")
print("=" * 60)

for chunk, metadata in app.stream(initial, stream_mode="messages"):
    node = metadata.get("langgraph_node", "?")
    # Only print AI token chunks (not ToolMessages or empty chunks)
    if chunk.type == "AIMessageChunk" and chunk.content:
        print(f"  [{node}] {chunk.content!r}", end="")

print("\n")  # newline after stream


# ── 4. Multiple stream modes simultaneously ───────────────────────────────────

print("=" * 60)
print("stream_mode=['updates','messages']  (combined)")
print("=" * 60)

for event in app.stream(initial, stream_mode=["updates", "messages"]):
    mode, data = event
    if mode == "updates":
        node_name = list(data.keys())[0]
        print(f"\n  [NODE COMPLETE] {node_name}")
    elif mode == "messages":
        chunk, meta = data
        if chunk.type == "AIMessageChunk" and chunk.content:
            print(f"  [TOKEN] {chunk.content!r}", end="", flush=True)
