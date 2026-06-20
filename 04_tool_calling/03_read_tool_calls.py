"""
Concept: Reading and dispatching response.tool_calls.

When the model wants to call a tool it sets response.tool_calls to a list
of dicts. Each dict has:

  {
    "name": "tool_name",        # which tool to call
    "args": {"arg1": "value"},  # arguments the model chose
    "id":   "call_abc123",      # unique ID — needed for ToolMessage (section 05)
    "type": "tool_call",
  }

Your code is responsible for:
  1. Reading the name and args.
  2. Looking up and executing the real function.
  3. Returning a ToolMessage with the result (section 05).

Run:
    python 04_tool_calling/03_read_tool_calls.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def get_date() -> str:
    """Return today's date in YYYY-MM-DD format."""
    from datetime import date
    return str(date.today())


tools = [add, multiply, get_date]
tool_map = {t.name: t for t in tools}   # name → callable lookup

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


# ── Helper: dispatch all tool calls in a response ─────────────────────────────

def dispatch_tool_calls(response) -> list[dict]:
    """Execute every tool call in the response and return the results."""
    results = []
    for tc in response.tool_calls:
        tool_fn = tool_map[tc["name"]]
        output  = tool_fn.invoke(tc["args"])
        results.append({
            "tool_call_id": tc["id"],
            "name":         tc["name"],
            "args":         tc["args"],
            "output":       output,
        })
    return results


# ── Example 1: single tool call ───────────────────────────────────────────────
response = llm.invoke([HumanMessage(content="What is 47 multiplied by 13?")])

print("=== Single tool call ===")
print("tool_calls raw:", response.tool_calls)
results = dispatch_tool_calls(response)
for r in results:
    print(f"  {r['name']}({r['args']}) → {r['output']}")


# ── Example 2: multiple tool calls in one response ───────────────────────────
response2 = llm.invoke([
    HumanMessage(content="What is 10 + 5, and also what is 4 * 7?")
])

print("\n=== Multiple tool calls ===")
for tc in response2.tool_calls:
    print(f"  Requested: {tc['name']}({tc['args']})")

results2 = dispatch_tool_calls(response2)
for r in results2:
    print(f"  Result   : {r['name']} → {r['output']}")


# ── Example 3: no-argument tool ───────────────────────────────────────────────
response3 = llm.invoke([HumanMessage(content="What is today's date?")])

print("\n=== No-argument tool ===")
results3 = dispatch_tool_calls(response3)
for r in results3:
    print(f"  {r['name']}() → {r['output']}")


# ── Example 4: no tool call (model answers directly) ─────────────────────────
response4 = llm.invoke([HumanMessage(content="What is the capital of France?")])

print("\n=== No tool call ===")
print(f"  tool_calls : {response4.tool_calls}")
print(f"  content    : {response4.content}")
