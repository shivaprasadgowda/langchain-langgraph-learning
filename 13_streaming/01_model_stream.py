"""
Concept: model.stream() — token-level streaming from the LLM.

Without streaming:
  response = llm.invoke(messages)     ← waits until all tokens are generated
  print(response.content)             ← entire text appears at once

With streaming:
  for chunk in llm.stream(messages):  ← yields one token (or a few) at a time
      print(chunk.content, end="")    ← text appears progressively

Why it matters for UX:
  A 200-token response at 50 tokens/sec takes 4 seconds.
  With streaming the user sees the first word after ~80ms instead of
  staring at a blank screen for 4 seconds.

Run:
    python 13_streaming/01_model_stream.py
"""

import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

messages = [
    SystemMessage(content="You are a helpful assistant. Be concise."),
    HumanMessage(content="Explain what a Kubernetes pod is in 3 sentences."),
]


# ── 1. Without streaming — entire response arrives at once ───────────────────

print("=== Without streaming ===")
start = time.time()
response = llm.invoke(messages)
elapsed = time.time() - start
print(f"Waited {elapsed:.2f}s for full response:")
print(response.content)


# ── 2. With streaming — tokens arrive progressively ───────────────────────────

print("\n=== With streaming (watch tokens appear) ===")
start = time.time()
first_token_time = None

for chunk in llm.stream(messages):
    if first_token_time is None and chunk.content:
        first_token_time = time.time() - start
    print(chunk.content, end="", flush=True)

total = time.time() - start
print(f"\n\nFirst token: {first_token_time:.2f}s  |  Total: {total:.2f}s")


# ── 3. Inspect chunk structure ────────────────────────────────────────────────

print("\n=== Chunk structure ===")
chunks = []
for chunk in llm.stream([HumanMessage(content="Say 'hello'")]):
    chunks.append(chunk)

print(f"Number of chunks  : {len(chunks)}")
print(f"First chunk type  : {type(chunks[0]).__name__}")
print(f"First chunk content: {chunks[0].content!r}")
print(f"Last chunk id     : {chunks[-1].id}")

# Reassemble the full message from chunks
from langchain_core.messages import AIMessageChunk
full = chunks[0]
for c in chunks[1:]:
    full = full + c   # AIMessageChunk supports + operator

print(f"Reassembled content: {full.content!r}")


# ── 4. Streaming through an LCEL chain ───────────────────────────────────────

print("\n=== Streaming through a chain (prompt | model | parser) ===")

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are concise."),
        ("human",  "{question}"),
    ])
    | llm
    | StrOutputParser()   # StrOutputParser extracts .content as a string
)

print("Answer: ", end="", flush=True)
for chunk in chain.stream({"question": "What is a Docker container in one sentence?"}):
    print(chunk, end="", flush=True)   # chunk is already a str thanks to StrOutputParser
print()


# ── 5. Collecting streamed output ─────────────────────────────────────────────

print("\n=== Collecting streamed output ===")
collected = []
for chunk in chain.stream({"question": "Name three Python data structures."}):
    collected.append(chunk)

full_text = "".join(collected)
print(f"Collected {len(collected)} chunks → {len(full_text)} chars")
print(f"Full text: {full_text}")
