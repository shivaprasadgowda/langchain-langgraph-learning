"""
Concept: The three core message types in LangChain.

Every conversation with a chat model is a list of messages.
LangChain defines three roles:

  SystemMessage  — sets the model's persona / instructions.
                   Sent once, usually at the top of the list.
                   The model never "replies" as a system.

  HumanMessage   — a turn from the user.

  AIMessage      — a turn from the model (returned by .invoke(),
                   but you can also add past AI turns to history).

Together they form the messages list that the model sees as context.

Run:
    python 02_messages_and_prompts/01_message_types.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Build a simple conversation ───────────────────────────────────────────────

messages = [
    # 1. System: tell the model who it is
    SystemMessage(content="You are a concise Python tutor. Keep answers to one sentence."),

    # 2. Human: first question
    HumanMessage(content="What is a list comprehension?"),
]

response = llm.invoke(messages)
print("AI:", response.content)

# ── Inspect message attributes ────────────────────────────────────────────────

print("\n--- Message attributes ---")
for msg in messages:
    # Every message has .content and .type
    print(f"type={msg.type!r:12}  content={msg.content!r}")

# The response is an AIMessage
print(f"type={response.type!r:12}  content={response.content!r}")

# ── All three types side by side ──────────────────────────────────────────────

print("\n--- All three types ---")
system = SystemMessage(content="You are helpful.")
human  = HumanMessage(content="Hello!")
ai     = AIMessage(content="Hi there! How can I help?")

for msg in [system, human, ai]:
    print(f"{msg.type}: {msg.content}")
