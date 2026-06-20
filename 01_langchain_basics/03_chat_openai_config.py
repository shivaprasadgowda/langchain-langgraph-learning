"""
Concept: Configuring ChatOpenAI — model, temperature, max_tokens.

Key parameters and when to change them:

  model         — which OpenAI model to use.
                  gpt-4o-mini : cheap, fast, good for most tasks.
                  gpt-4o      : more capable, higher cost.

  temperature   — randomness of the output (0.0 – 2.0).
                  0.0 : deterministic, same answer every time.
                        Use for structured output, classification, code.
                  0.7 : default — balanced creativity.
                  1.0+: more varied / creative, less predictable.

  max_tokens    — hard cap on output length (tokens ≈ words * 1.3).
                  Prevents runaway responses and controls cost.

  timeout       — seconds before the SDK raises a timeout error.
                  Important in production to avoid hanging requests.

Run:
    python 01_langchain_basics/03_chat_openai_config.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

prompt = [HumanMessage(content="Give me a creative name for a Python AI library.")]

# ── 1. Deterministic output (temperature=0) ──────────────────────────────────
deterministic = ChatOpenAI(model="gpt-4o-mini", temperature=0)
r1 = deterministic.invoke(prompt)
print("temperature=0  :", r1.content)

# Run it a second time — the answer should be identical.
r1b = deterministic.invoke(prompt)
print("temperature=0  :", r1b.content, "(same?", r1.content == r1b.content, ")")

# ── 2. Creative output (temperature=1) ───────────────────────────────────────
creative = ChatOpenAI(model="gpt-4o-mini", temperature=1)
r2 = creative.invoke(prompt)
print("temperature=1  :", r2.content)

# ── 3. Limited output length ──────────────────────────────────────────────────
# max_tokens=10 forces a very short reply; the model may cut off mid-sentence.
brief = ChatOpenAI(model="gpt-4o-mini", max_tokens=10)
r3 = brief.invoke([HumanMessage(content="Explain machine learning.")])
print("max_tokens=10  :", r3.content)
print("finish_reason  :", r3.response_metadata.get("finish_reason"))
# finish_reason will be "length" (truncated) instead of "stop" (complete).

# ── 4. Combining parameters ───────────────────────────────────────────────────
production = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,    # deterministic for reliable structured responses
    max_tokens=256,   # cap output
    timeout=30,       # fail fast if the API is slow
)
r4 = production.invoke([HumanMessage(content="What is a REST API?")])
print("production cfg :", r4.content[:120], "...")
