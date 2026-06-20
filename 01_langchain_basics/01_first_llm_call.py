"""
Concept: First LLM call with LangChain.

Why LangChain instead of the raw OpenAI SDK?
- The raw SDK returns provider-specific objects (openai.types.chat.ChatCompletion).
- LangChain wraps every model behind a common interface (BaseChatModel).
- The same code runs against OpenAI, Anthropic, Google, or a local model by
  swapping one import — no logic changes needed.
- It also unlocks chains, agents, and tools that are built on that interface.

Run:
    python 01_langchain_basics/01_first_llm_call.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load OPENAI_API_KEY from .env at the repo root
load_dotenv()

# ChatOpenAI is LangChain's wrapper around the OpenAI chat endpoint.
# Defaults: model="gpt-4o-mini", temperature=0.7
llm = ChatOpenAI(model="gpt-4o-mini")

# Build the message list. HumanMessage = a turn from the user.
messages = [HumanMessage(content="What is LangChain in one sentence?")]

# .invoke() sends the messages and returns an AIMessage.
response = llm.invoke(messages)

print(response.content)
