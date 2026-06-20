"""
Concept: Managing chat history manually.

A chat model has no memory between calls — it only knows what is in the
messages list you send. To give it memory you must:
  1. Start with a system message.
  2. Append each HumanMessage before calling the model.
  3. Append the returned AIMessage to the history.
  4. Send the full list on every turn.

This is the foundation that LangGraph's checkpointer automates (section 08).

Run:
    python 02_messages_and_prompts/02_manual_chat_history.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Initialise history with a system prompt ───────────────────────────────────
history = [
    SystemMessage(content="You are a helpful assistant who remembers what the user tells you."),
]

def chat(user_input: str) -> str:
    # Add the new human turn
    history.append(HumanMessage(content=user_input))

    # Send the full history so the model has all context
    response = llm.invoke(history)

    # Store the model's reply so it appears in the next call
    history.append(response)

    return response.content


# ── Simulate a multi-turn conversation ───────────────────────────────────────
print("Turn 1:", chat("My name is Alex and I'm learning LangChain."))
print("Turn 2:", chat("What's my name?"))
print("Turn 3:", chat("What am I learning?"))

# ── Show the accumulated history ──────────────────────────────────────────────
print("\n--- Full history ---")
for i, msg in enumerate(history):
    print(f"[{i}] {msg.type:8} : {msg.content[:80]}")
