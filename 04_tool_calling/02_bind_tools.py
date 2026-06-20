"""
Concept: Binding tools to the model with bind_tools.

llm.bind_tools([tool1, tool2]) returns a new Runnable that includes the
tool schemas in every request. The model can then choose to call a tool
instead of (or as well as) writing a text reply.

Key behaviours:
  - The model may call zero, one, or multiple tools in one response.
  - If it calls a tool, response.content is usually empty.
  - If it answers directly, response.tool_calls is [].
  - tool_choice="any" forces the model to always pick a tool.
  - tool_choice="none" prevents tool calling entirely.

Run:
    python 04_tool_calling/02_bind_tools.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()


# ── Define a small tool set ───────────────────────────────────────────────────

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol (e.g. AAPL, GOOGL)."""
    prices = {"AAPL": 189.50, "GOOGL": 175.20, "MSFT": 415.00}
    price = prices.get(ticker.upper(), "unknown")
    return f"{ticker.upper()}: ${price}"


@tool
def get_company_info(ticker: str) -> str:
    """Get basic company information for a ticker symbol."""
    info = {
        "AAPL": "Apple Inc. — consumer electronics and software.",
        "GOOGL": "Alphabet Inc. — search, cloud, and advertising.",
        "MSFT": "Microsoft Corp. — software, cloud, and hardware.",
    }
    return info.get(ticker.upper(), f"No info found for {ticker}.")


tools = [get_stock_price, get_company_info]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# bind_tools sends the tool schemas with every request from now on
llm_with_tools = llm.bind_tools(tools)


# ── Case 1: message that triggers a tool call ─────────────────────────────────
response = llm_with_tools.invoke([HumanMessage(content="What is Apple's stock price?")])

print("=== Tool call triggered ===")
print("content    :", repr(response.content))       # usually empty
print("tool_calls :", response.tool_calls)
# [{'name': 'get_stock_price', 'args': {'ticker': 'AAPL'}, 'id': '...', 'type': 'tool_call'}]


# ── Case 2: message that does NOT trigger a tool call ─────────────────────────
response2 = llm_with_tools.invoke([HumanMessage(content="What is 2 + 2?")])

print("\n=== No tool call ===")
print("content    :", response2.content)
print("tool_calls :", response2.tool_calls)   # []


# ── Case 3: force a tool call with tool_choice="any" ─────────────────────────
llm_forced = llm.bind_tools(tools, tool_choice="any")
response3 = llm_forced.invoke([HumanMessage(content="Hello!")])

print("\n=== Forced tool call (tool_choice='any') ===")
print("tool_calls :", response3.tool_calls)


# ── Inspect the bound tool schemas sent to the API ───────────────────────────
print("\n=== Bound tool names ===")
for t in tools:
    print(" -", t.name)
