"""
Concept: Full manual agent loop — no framework.

The loop runs until the model stops requesting tools (tool_calls is empty),
at which point its response is the final natural-language answer.

Loop shape:
  while True:
      response = llm.invoke(messages)
      if no tool_calls → break (final answer)
      for each tool_call:
          execute tool
          append ToolMessage to history
      append AIMessage to history
      continue

This is exactly what LangGraph's ToolNode + tools_condition automates
in section 10. Building it by hand makes that abstraction concrete.

Run:
    python 05_manual_agent_loop/02_manual_loop.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()


# ── Tools (reuse from section 04 pattern) ─────────────────────────────────────

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
    return f"The weather in {city} is 22°C and sunny."


@tool
def search_jira(query: str) -> str:
    """Search Jira tickets by keyword. Returns matching ticket IDs and titles."""
    return f"Found tickets for '{query}': PROJ-101 (Login bug), PROJ-202 (API timeout)"


tools = [add, multiply, get_weather, search_jira]
tool_map = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(user_input: str) -> str:
    """Run the manual agent loop and return the final answer."""
    messages = [
        SystemMessage(content="You are a helpful assistant with access to maths, weather, and Jira tools."),
        HumanMessage(content=user_input),
    ]

    step = 0
    while True:
        step += 1
        print(f"  [step {step}] calling model...")
        response = llm.invoke(messages)

        # Append the AIMessage to history regardless of whether it has tool calls
        messages.append(response)

        # No tool calls → model has its final answer
        if not response.tool_calls:
            print(f"  [step {step}] model finished — no more tool calls")
            return response.content

        # Execute every tool the model requested (may be parallel)
        print(f"  [step {step}] model requested {len(response.tool_calls)} tool call(s)")
        for tc in response.tool_calls:
            print(f"           → {tc['name']}({tc['args']})")
            result = tool_map[tc["name"]].invoke(tc["args"])
            print(f"           ← {result}")
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc["id"],
            ))


# ── Test cases ────────────────────────────────────────────────────────────────

queries = [
    # Single tool, one step
    "What's the weather in London?",
    # Multi-step: model may chain arithmetic
    "What is (15 + 7) multiplied by 3?",
    # Multi-tool in one round
    "What's the weather in Paris and also search Jira for 'deployment'?",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"User: {q}")
    answer = run_agent(q)
    print(f"Final answer: {answer}")
