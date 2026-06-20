"""
Concept: Inspecting the AIMessage response object.

Every LangChain chat model returns an AIMessage.
Knowing its fields is essential for:
  - Extracting the text reply (.content)
  - Detecting tool calls (.tool_calls) — used in section 04
  - Tracking token usage (.usage_metadata)
  - Reading raw provider metadata (.response_metadata)

Run:
    python 01_langchain_basics/02_ai_message_response.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke([HumanMessage(content="Name one benefit of Python.")])

# ── Core fields ──────────────────────────────────────────────────────────────

# The text the model generated
print("content        :", response.content)

# Always "ai" for model responses
print("type           :", response.type)

# Unique ID assigned by the provider for this completion
print("id             :", response.id)

# ── Token usage ──────────────────────────────────────────────────────────────
# usage_metadata is a standard LangChain field populated by most providers.
# Useful for cost tracking and staying within context limits.
meta = response.usage_metadata
if meta:
    print("input_tokens   :", meta["input_tokens"])
    print("output_tokens  :", meta["output_tokens"])
    print("total_tokens   :", meta["total_tokens"])

# ── Provider metadata ────────────────────────────────────────────────────────
# response_metadata contains the raw provider response fields,
# e.g. finish_reason, model name, system fingerprint.
print("finish_reason  :", response.response_metadata.get("finish_reason"))
print("model          :", response.response_metadata.get("model_name"))

# ── Tool calls (preview) ─────────────────────────────────────────────────────
# Will be non-empty only when tools are bound to the model (section 04).
print("tool_calls     :", response.tool_calls)  # [] here
