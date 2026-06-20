"""
Concept: Manual agent loop (section 05) vs ToolNode agent (section 10).

Both approaches solve the same problem — run an agent that can call tools
and loop until the model is done. The code below runs the same question
through both implementations so the difference is immediately visible.

Read top-to-bottom: the manual loop is verbose boilerplate; the LangGraph
version is a thin graph definition that delegates the mechanics.

Run:
    python 10_toolnode_agents/04_manual_vs_toolnode.py
"""

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ── Shared tools ──────────────────────────────────────────────────────────────

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
    return f"The weather in {city} is 21°C and sunny."


tools     = [add, multiply, get_weather]
tool_map  = {t.name: t for t in tools}
QUESTION  = "What is (12 + 8) multiplied by 5, and what's the weather in Berlin?"
SYSTEM    = SystemMessage(content="You are a helpful assistant with maths and weather tools.")


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 1: Manual agent loop (section 05 pattern)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("APPROACH 1: Manual agent loop")
print("=" * 60)

llm_plain = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def run_manual_agent(question: str, max_steps: int = 8) -> str:
    messages = [SYSTEM, HumanMessage(content=question)]

    for step in range(1, max_steps + 1):
        response = llm_plain.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        print(f"  [step {step}] {len(response.tool_calls)} tool call(s):")
        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            print(f"    {tc['name']}({tc['args']}) → {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return "[max_steps reached]"

answer_manual = run_manual_agent(QUESTION)
print(f"\nAnswer: {answer_manual}")

print("""
Lines of agent logic written:
  - bind_tools on llm                          ✓
  - while/for loop                             ✓ (manual)
  - response.tool_calls check                  ✓ (manual)
  - tool_map dispatch                          ✓ (manual)
  - ToolMessage construction with correct id   ✓ (manual)
  - messages.append for AIMessage + ToolMsg    ✓ (manual)
  - max_steps guard                            ✓ (manual)
  - NO persistence across calls
  - NO streaming hooks
  - NO human-in-the-loop support
""")


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 2: LangGraph ToolNode agent
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("APPROACH 2: LangGraph ToolNode agent")
print("=" * 60)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm_graph = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def llm_node(state: State) -> dict:
    return {"messages": [llm_graph.invoke([SYSTEM] + state["messages"])]}

builder = StateGraph(State)
builder.add_node("llm",   llm_node)
builder.add_node("tools", ToolNode(tools))   # ← replaces all manual dispatch logic
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)  # ← replaces manual check
builder.add_edge("tools", "llm")

app = builder.compile()

result = app.invoke({"messages": [HumanMessage(content=QUESTION)]})
answer_graph = result["messages"][-1].content
print(f"Answer: {answer_graph}")

print("""
Lines of agent logic written:
  - bind_tools on llm                          ✓
  - ToolNode(tools)                            ✓ (one line replaces dispatch loop)
  - tools_condition                            ✓ (one line replaces manual check)
  - add_edge("tools", "llm")                  ✓ (one line replaces loop structure)
  - Persistence: add checkpointer= to compile()
  - Streaming:   app.stream() — built in
  - Human-in-the-loop: interrupt_before= — built in
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Side-by-side comparison table
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("COMPARISON TABLE")
print("=" * 60)

rows = [
    ("Tool dispatch",         "tool_map + for loop",      "ToolNode(tools)"),
    ("Loop control",          "while True + break",        "tools_condition edge"),
    ("ToolMessage creation",  "manual, per tool call",     "automatic"),
    ("Parallel tool calls",   "manual nested loop",        "automatic"),
    ("Persistence",           "not supported",             "compile(checkpointer=...)"),
    ("Streaming",             "custom callbacks",          "app.stream()"),
    ("Human-in-the-loop",     "complex async logic",       "interrupt_before="),
    ("Testing nodes",         "test entire function",      "test each node in isolation"),
    ("Max steps guard",       "manual",                    "recursion_limit in config"),
    ("LangSmith tracing",     "partial",                   "full per-node traces"),
]

print(f"\n  {'Feature':<28} {'Manual loop':<28} {'ToolNode (LangGraph)'}")
print("  " + "─" * 80)
for feature, manual, lg in rows:
    print(f"  {feature:<28} {manual:<28} {lg}")
