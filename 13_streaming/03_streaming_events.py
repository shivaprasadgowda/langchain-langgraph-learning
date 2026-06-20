"""
Concept: astream_events() — fine-grained async event stream.

astream_events() is the most granular streaming API. It emits events for:
  - Individual LLM token chunks  (on_chat_model_stream)
  - Tool start / end             (on_tool_start, on_tool_end)
  - Chain / node start / end     (on_chain_start, on_chain_end)
  - Retriever calls              (on_retriever_start, on_retriever_end)

Each event is a dict:
  {
    "event":  "on_chat_model_stream",
    "name":   "ChatOpenAI",
    "data":   {"chunk": AIMessageChunk(...)},
    "tags":   [...],
    "run_id": "...",
    "metadata": {"langgraph_node": "llm", ...},
  }

This is the API used by production UIs to show:
  - Streaming tokens in real time
  - "Agent is searching Jira..." progress indicators
  - Tool call arguments before they execute

This file uses asyncio because astream_events is an async generator.

Run:
    python 13_streaming/03_streaming_events.py
"""

import asyncio
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


# ── Graph ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def search_jira(query: str) -> str:
    """Search Jira tickets by keyword."""
    return f"[JIRA] Results for '{query}': PROJ-101 (Login bug), PROJ-202 (API timeout)"


@tool
def get_aws_cost(month: str) -> str:
    """Get AWS cost summary for a given month (YYYY-MM)."""
    return f"[AWS] {month}: EC2 $312, RDS $98, S3 $15, Total $425"


tools = [search_jira, get_aws_cost]
llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def llm_node(state: State) -> dict:
    system = SystemMessage(content="You are a DevOps assistant with Jira and AWS tools.")
    return {"messages": [llm.invoke([system] + state["messages"])]}

builder = StateGraph(State)
builder.add_node("llm",   llm_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")

app = builder.compile()

QUESTION = "Search Jira for login bugs and also get AWS cost for June 2026."


# ── 1. Stream ALL events ──────────────────────────────────────────────────────

async def demo_all_events() -> None:
    print("=" * 60)
    print("ALL events (first 15)")
    print("=" * 60)
    count = 0
    async for event in app.astream_events(
        {"messages": [HumanMessage(content=QUESTION)]},
        version="v2",
    ):
        if count >= 15:
            print("  ... (truncated)")
            break
        node = event.get("metadata", {}).get("langgraph_node", "—")
        print(f"  [{count:02d}] event={event['event']:<30} name={event['name']:<20} node={node}")
        count += 1


# ── 2. Filter: LLM token chunks only ──────────────────────────────────────────

async def demo_token_stream() -> None:
    print("\n" + "=" * 60)
    print("LLM TOKENS ONLY  (on_chat_model_stream)")
    print("=" * 60)

    print("Answer: ", end="", flush=True)
    async for event in app.astream_events(
        {"messages": [HumanMessage(content=QUESTION)]},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # Only print text tokens from the final answer (not tool-call generation)
            if chunk.content and not chunk.tool_call_chunks:
                print(chunk.content, end="", flush=True)
    print()


# ── 3. Filter: tool lifecycle events ──────────────────────────────────────────

async def demo_tool_events() -> None:
    print("\n" + "=" * 60)
    print("TOOL EVENTS  (on_tool_start + on_tool_end)")
    print("=" * 60)
    async for event in app.astream_events(
        {"messages": [HumanMessage(content=QUESTION)]},
        version="v2",
    ):
        if event["event"] == "on_tool_start":
            print(f"  ▶ TOOL START: {event['name']}")
            print(f"    input: {event['data'].get('input', {})}")
        elif event["event"] == "on_tool_end":
            output = event["data"].get("output", "")
            # output may be a ToolMessage or string
            content = output.content if hasattr(output, "content") else str(output)
            print(f"  ■ TOOL END  : {event['name']} → {content[:60]}")


# ── 4. Filter: node lifecycle events ──────────────────────────────────────────

async def demo_node_events() -> None:
    print("\n" + "=" * 60)
    print("NODE LIFECYCLE  (on_chain_start + on_chain_end per node)")
    print("=" * 60)
    async for event in app.astream_events(
        {"messages": [HumanMessage(content=QUESTION)]},
        version="v2",
    ):
        node = event.get("metadata", {}).get("langgraph_node")
        # langgraph_node is set on events emitted from within a named node
        if node and event["event"] in ("on_chain_start", "on_chain_end"):
            symbol = "▶" if "start" in event["event"] else "■"
            print(f"  {symbol} {event['event']:<20} node={node!r}")


# ── 5. Build a UI event stream (what a frontend would consume) ────────────────

async def demo_ui_events() -> None:
    """Produce a simplified event stream suitable for a chat UI."""
    print("\n" + "=" * 60)
    print("UI EVENT STREAM  (what a frontend would receive)")
    print("=" * 60)

    async for event in app.astream_events(
        {"messages": [HumanMessage(content=QUESTION)]},
        version="v2",
    ):
        ev   = event["event"]
        node = event.get("metadata", {}).get("langgraph_node", "")

        if ev == "on_tool_start" and node == "tools":
            print(f"  UI ← {{'type': 'tool_start', 'name': {event['name']!r}}}")

        elif ev == "on_tool_end" and node == "tools":
            out = event["data"].get("output", "")
            content = out.content if hasattr(out, "content") else str(out)
            print(f"  UI ← {{'type': 'tool_end', 'result': {content[:40]!r}}}")

        elif ev == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content and not chunk.tool_call_chunks:
                print(f"  UI ← {{'type': 'token', 'text': {chunk.content!r}}}", end="\r")

    print("\n  UI ← {'type': 'done'}")


# ── Run all demos ─────────────────────────────────────────────────────────────

async def main() -> None:
    await demo_all_events()
    await demo_token_stream()
    await demo_tool_events()
    await demo_node_events()
    await demo_ui_events()


asyncio.run(main())
