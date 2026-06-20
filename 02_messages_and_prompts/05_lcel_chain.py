"""
Concept: LCEL chain — prompt | model | parser.

LCEL (LangChain Expression Language) composes Runnables with the | operator.
Each component implements .invoke() / .stream() / .batch() so the interface
is uniform regardless of what sits in each slot.

Typical chain shape:
    prompt | model | output_parser

  prompt        — ChatPromptTemplate  → produces a list of messages
  model         — ChatOpenAI          → produces an AIMessage
  output_parser — StrOutputParser     → extracts .content as a plain string
                  JsonOutputParser    → parses JSON from .content
                  PydanticOutputParser→ validates against a Pydantic model

Why add a parser?
  Without one, the chain returns an AIMessage object.
  StrOutputParser extracts .content so callers get a plain string — no need
  to know about LangChain internals.

Run:
    python 02_messages_and_prompts/05_lcel_chain.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Step 1: Without a parser — returns AIMessage ──────────────────────────────
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    ("human", "{question}"),
])

chain_no_parser = template | llm
result_raw = chain_no_parser.invoke({"question": "What is a Python decorator?"})
print("Without parser →", type(result_raw).__name__)  # AIMessage
print("  .content:", result_raw.content[:80])

# ── Step 2: With StrOutputParser — returns plain str ─────────────────────────
chain = template | llm | StrOutputParser()
result_str = chain.invoke({"question": "What is a Python decorator?"})
print("\nWith parser    →", type(result_str).__name__)  # str
print(" ", result_str[:80])

# ── Step 3: .batch() — invoke the chain on multiple inputs at once ────────────
questions = [
    {"question": "What is a list?"},
    {"question": "What is a dict?"},
    {"question": "What is a set?"},
]
results = chain.batch(questions)
print("\n--- Batch results ---")
for q, r in zip(questions, results):
    print(f"Q: {q['question']}")
    print(f"A: {r[:80]}\n")

# ── Step 4: .stream() — receive tokens as they are generated ─────────────────
print("--- Streaming ---")
for chunk in chain.stream({"question": "Name three Python built-in functions."}):
    print(chunk, end="", flush=True)
print()  # newline after stream ends
