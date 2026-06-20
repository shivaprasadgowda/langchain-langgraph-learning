"""
Concept: A self-contained calculator tool example.

Shows the complete one-round flow:
  1. User asks a maths question.
  2. Model returns a tool_call with the operator and operands.
  3. Code executes the calculation and prints the result.

This is intentionally one round only — no ToolMessage feedback loop.
The full multi-round agent loop (where the result is fed back to the
model for a final natural-language answer) is built in section 05.

Run:
    python 04_tool_calling/04_calculator_tool.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b and return the result. b must not be zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent


calculator_tools = [add, subtract, multiply, divide, power]
tool_map = {t.name: t for t in calculator_tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(calculator_tools)

messages = [
    SystemMessage(content="You are a calculator assistant. Use the available tools to answer maths questions."),
]


# ── Helper ────────────────────────────────────────────────────────────────────

def calculate(question: str) -> None:
    response = llm.invoke(messages + [HumanMessage(content=question)])

    if not response.tool_calls:
        print(f"Q: {question}")
        print(f"A: {response.content}  (answered directly, no tool needed)")
        return

    for tc in response.tool_calls:
        fn     = tool_map[tc["name"]]
        result = fn.invoke(tc["args"])
        print(f"Q: {question}")
        print(f"   Tool  : {tc['name']}({tc['args']})")
        print(f"   Result: {result}")


# ── Test cases ────────────────────────────────────────────────────────────────

calculate("What is 128 divided by 4?")
calculate("Multiply 17 by 23.")
calculate("What is 2 to the power of 10?")
calculate("Subtract 99 from 256.")
calculate("What is 1000 plus 337?")
