"""
Concept: Text splitting — chunking documents for embedding.

LLMs have context limits and embedding models have token limits (~8k tokens).
A 100-page PDF must be split into small, overlapping chunks before embedding.

Why overlap?
  If a key sentence falls at the boundary between chunk 3 and chunk 4,
  an overlap ensures it appears fully in at least one chunk. Without
  overlap, boundary sentences get cut off and become unretrievable.

RecursiveCharacterTextSplitter is the recommended default:
  It tries to split on paragraph breaks (\n\n), then sentences (\n),
  then words (" "), and finally characters — using the first separator
  that produces chunks smaller than chunk_size.

Run:
    python 09_rag/02_text_splitting.py
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Sample document (long enough to require splitting) ────────────────────────

long_doc = Document(
    page_content="""
LangGraph Overview

LangGraph is a library for building stateful, multi-step applications with large
language models. It was created by the LangChain team to address the limitations
of simple chain-based architectures when building agents.

Core Concepts

The fundamental unit in LangGraph is the StateGraph. A StateGraph is defined by
a state schema — a TypedDict that all nodes read from and write to. Every node is
a plain Python function that accepts the current state and returns a partial update.
LangGraph merges the update into the state using registered reducers.

Nodes and Edges

Nodes represent discrete units of computation: calling an LLM, running a tool,
classifying input, or writing to a database. Edges define the flow of control.
A normal edge always routes to the same next node. A conditional edge calls a
routing function on the current state and uses the returned string to decide
which node to run next.

Persistence

LangGraph supports pluggable checkpointers that snapshot state after every node.
MemorySaver stores state in a Python dict and is suitable for development.
AsyncPostgresSaver persists state to a PostgreSQL database and is recommended
for production deployments where durability and horizontal scaling are required.

Human-in-the-Loop

LangGraph provides an interrupt mechanism that pauses graph execution at a
specified node and suspends the process until human input is provided. This
enables approval workflows where an agent must get confirmation before taking
a destructive or irreversible action such as deleting a resource or sending
an email to a customer.
""".strip(),
    metadata={"source": "docs/langgraph_overview.txt"},
)


# ── 1. Default splitter ───────────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # max characters per chunk
    chunk_overlap=50,     # characters repeated at chunk boundaries
    length_function=len,  # measure in characters (use tiktoken for tokens)
)

chunks = splitter.split_documents([long_doc])

print(f"=== Split result ===")
print(f"Original length : {len(long_doc.page_content)} chars")
print(f"Chunks produced : {len(chunks)}")
print(f"chunk_size      : 300  chunk_overlap: 50")

for i, chunk in enumerate(chunks):
    print(f"\n[chunk {i}] {len(chunk.page_content)} chars  source={chunk.metadata['source']}")
    print(f"  {chunk.page_content[:120]}...")


# ── 2. Effect of overlap — show boundary repetition ──────────────────────────

print("\n=== Overlap at chunk boundary ===")
if len(chunks) >= 2:
    end_of_0   = chunks[0].page_content[-60:]
    start_of_1 = chunks[1].page_content[:60:]
    print(f"End of chunk 0   : ...{end_of_0!r}")
    print(f"Start of chunk 1 : {start_of_1!r}...")


# ── 3. Metadata is preserved on every chunk ───────────────────────────────────

print("\n=== Metadata preserved ===")
for chunk in chunks[:3]:
    print(f"  {chunk.metadata}")


# ── 4. Chunk size comparison ──────────────────────────────────────────────────

print("\n=== Chunk count by size ===")
for size in [100, 300, 500, 1000]:
    s = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=20)
    n = len(s.split_documents([long_doc]))
    print(f"  chunk_size={size:<5} → {n} chunks")
