"""
Concept: ToolMessage — feeding tool results back to the LLM.

After your code executes a tool, the result must be sent back to the
model as a ToolMessage. Without it the model doesn't know what the tool
returned and cannot produce a grounded final answer.

ToolMessage requires:
  content      — the tool's return value (as a string).
  tool_call_id — the id from the original tool_call dict. This is how
                 the model matches the result to the request it made.

Message order the model expects:
  [SystemMessage]
  [HumanMessage]          ← user turn
  [AIMessage]             ← model's tool_call request (content may be empty)
  [ToolMessage]           ← your tool result
  ... (repeat for each tool call)
  [AIMessage]             ← model's final natural-language answer

Run:
    python 05_manual_agent_loop/01_tool_message.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 18°C and cloudy."


tools = [get_weather]
tool_map = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


# ── Step 1: first model call — model requests a tool ─────────────────────────

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What's the weather like in Tokyo?"),
]

ai_message = llm.invoke(messages)
print("=== Step 1: AIMessage (tool request) ===")
print("content    :", repr(ai_message.content))   # usually empty
print("tool_calls :", ai_message.tool_calls)

# The AIMessage itself must be appended to history before adding ToolMessage
messages.append(ai_message)


# ── Step 2: execute the tool and build a ToolMessage ─────────────────────────

tc = ai_message.tool_calls[0]
tool_result = tool_map[tc["name"]].invoke(tc["args"])

tool_message = ToolMessage(
    content=str(tool_result),       # must be a string
    tool_call_id=tc["id"],          # links result to the request
)

print("\n=== Step 2: ToolMessage ===")
print("content      :", tool_message.content)
print("tool_call_id :", tool_message.tool_call_id)
print("type         :", tool_message.type)   # "tool"

messages.append(tool_message)


# ── Step 3: second model call — model reads the result and answers ────────────

final = llm.invoke(messages)
print("\n=== Step 3: Final AIMessage ===")
print("content    :", final.content)
print("tool_calls :", final.tool_calls)   # [] — model is done with tools

print("\n=== Full message history ===")
for i, m in enumerate(messages + [final]):
    preview = m.content[:60] if m.content else repr(m.tool_calls)
    print(f"[{i}] {m.type:10} : {preview}")
