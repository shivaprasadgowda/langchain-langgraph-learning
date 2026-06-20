"""
Concept: MessagesPlaceholder — inject dynamic history into a template.

A ChatPromptTemplate with a MessagesPlaceholder has a "slot" where you
pass an actual list of messages at runtime. This is how LangGraph and
memory systems plug conversation history into a fixed prompt structure.

Without MessagesPlaceholder you would have to rebuild the template every
time the history grows — that defeats the purpose of a template.

Run:
    python 02_messages_and_prompts/04_messages_placeholder.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Template with a history slot ──────────────────────────────────────────────
# "chat_history" is the variable name — pass a list of messages under that key.
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}"),
])

# ── First turn — history is empty ────────────────────────────────────────────
history = []

chain = template | llm

response1 = chain.invoke({
    "chat_history": history,
    "user_input": "My favourite language is Python.",
})
print("Turn 1:", response1.content)

# Accumulate history
history.append(HumanMessage(content="My favourite language is Python."))
history.append(response1)

# ── Second turn — history carries the earlier exchange ───────────────────────
response2 = chain.invoke({
    "chat_history": history,
    "user_input": "What's my favourite language?",
})
print("Turn 2:", response2.content)

# ── Show what the model actually received on turn 2 ──────────────────────────
print("\n--- Messages sent on turn 2 ---")
filled = template.format_messages(
    chat_history=history,
    user_input="What's my favourite language?",
)
for msg in filled:
    print(f"  {msg.type:8}: {msg.content}")
