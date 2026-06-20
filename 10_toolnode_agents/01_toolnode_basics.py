"""
Concept: ToolNode — the managed tool dispatcher.

ToolNode is a pre-built LangGraph node that implements the tool-dispatch
logic from section 05 (manual loop) as a reusable graph node.

What ToolNode does internally (simplified):
    1. Reads the last AIMessage from state["messages"].
    2. For every tool_call in that message, looks up and invokes the function.
    3. Wraps each result in a ToolMessage with the correct tool_call_id.
    4. Returns {"messages": [ToolMessage, ToolMessage, ...]} as a state update.

Parallel tool calls are handled automatically — if the model requests
two tools at once, ToolNode runs them concurrently.

Run:
    python 10_toolnode_agents/01_toolnode_basics.py
"""

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

load_dotenv()


# ── Define tools ──────────────────────────────────────────────────────────────

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 18°C and partly cloudy."


tools = [add, multiply, get_weather]

# ── Create ToolNode ───────────────────────────────────────────────────────────
# Pass the list of tools — ToolNode builds its own internal tool_map.

tool_node = ToolNode(tools)


# ── Demonstrate ToolNode directly (no graph needed) ───────────────────────────
# ToolNode expects state with a "messages" key where the last message
# is an AIMessage containing tool_calls.

print("=== Single tool call ===")

# Simulate an AIMessage the LLM would produce
ai_msg_single = AIMessage(
    content="",
    tool_calls=[{
        "name": "get_weather",
        "args": {"city": "Tokyo"},
        "id":   "call_001",
        "type": "tool_call",
    }]
)

result = tool_node.invoke({"messages": [ai_msg_single]})

for msg in result["messages"]:
    print(f"  type           : {msg.type}")
    print(f"  tool_call_id   : {msg.tool_call_id}")
    print(f"  content        : {msg.content}")


# ── Parallel tool calls ───────────────────────────────────────────────────────
print("\n=== Parallel tool calls (model requests two at once) ===")

ai_msg_parallel = AIMessage(
    content="",
    tool_calls=[
        {"name": "add",      "args": {"a": 15, "b": 27}, "id": "call_002", "type": "tool_call"},
        {"name": "multiply", "args": {"a": 6,  "b": 7},  "id": "call_003", "type": "tool_call"},
    ]
)

result_parallel = tool_node.invoke({"messages": [ai_msg_parallel]})

for msg in result_parallel["messages"]:
    print(f"  tool_call_id={msg.tool_call_id}  result={msg.content}")


# ── ToolNode raises on unknown tool name ──────────────────────────────────────
print("\n=== Unknown tool — KeyError ===")
ai_msg_bad = AIMessage(
    content="",
    tool_calls=[{
        "name": "nonexistent_tool",
        "args": {},
        "id":   "call_004",
        "type": "tool_call",
    }]
)

try:
    tool_node.invoke({"messages": [ai_msg_bad]})
except Exception as e:
    print(f"  Caught {type(e).__name__}: {e}")


# ── Inspect ToolNode internals ────────────────────────────────────────────────
print("\n=== ToolNode tool names ===")
for name in tool_node.tools_by_name:
    print(f"  {name}")
