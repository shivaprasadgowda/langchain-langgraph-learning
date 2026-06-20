"""
Concept: ChatPromptTemplate — reusable prompt with variables.

Instead of building message lists by hand every time, a ChatPromptTemplate
lets you define the structure once and fill in variables at call time.

Benefits:
  - Reuse the same prompt shape across different inputs.
  - Keep prompts in one place, not scattered across call sites.
  - Compose with models and parsers via LCEL ( | ).

Run:
    python 02_messages_and_prompts/03_chat_prompt_template.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Define a reusable template ────────────────────────────────────────────────
# Variables are wrapped in {curly_braces}.
# "system" and "human" are role shortcuts recognised by ChatPromptTemplate.
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Answer in {language}."),
    ("human", "{question}"),
])

# ── Inspect the template ──────────────────────────────────────────────────────
print("Input variables:", template.input_variables)
# ['domain', 'language', 'question']

# ── Format — produces a list of messages with variables substituted ───────────
filled = template.format_messages(
    domain="machine learning",
    language="plain English",
    question="What is overfitting?",
)
for msg in filled:
    print(f"{msg.type}: {msg.content}")

# ── Invoke directly via LCEL ( | ) ────────────────────────────────────────────
chain = template | llm

response = chain.invoke({
    "domain": "machine learning",
    "language": "plain English",
    "question": "What is overfitting?",
})
print("\nResponse:", response.content)

# ── Reuse the same template with different inputs ─────────────────────────────
response2 = chain.invoke({
    "domain": "databases",
    "language": "simple terms",
    "question": "What is an index?",
})
print("Response2:", response2.content)
