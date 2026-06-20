"""
Concept: Retriever — the standard interface for fetching relevant documents.

A retriever wraps a vector store and exposes a single method:
    retriever.invoke("query string") -> list[Document]

Why use a retriever instead of calling vectorstore.similarity_search() directly?
  - Retriever is a Runnable — it plugs into LCEL chains with |
  - It decouples the RAG chain from the underlying store
  - You can swap FAISS for Pinecone/pgvector without changing the chain

Configuration options:
  search_type="similarity"           — plain cosine/L2 similarity (default)
  search_type="mmr"                  — Maximal Marginal Relevance: diverse results
  search_type="similarity_score_threshold" — only return docs above a score
  search_kwargs={"k": 4}             — number of documents to return

Run:
    python 09_rag/04_retriever.py
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# ── Build a small vector store ────────────────────────────────────────────────

docs = [
    Document(page_content="LangGraph builds agent workflows as directed graphs.",
             metadata={"source": "lg_intro",   "page": 1}),
    Document(page_content="StateGraph defines the state schema as a TypedDict.",
             metadata={"source": "lg_state",   "page": 2}),
    Document(page_content="Nodes are Python functions: (state) -> dict.",
             metadata={"source": "lg_nodes",   "page": 3}),
    Document(page_content="Conditional edges call a routing function on state.",
             metadata={"source": "lg_edges",   "page": 4}),
    Document(page_content="MemorySaver checkpointer stores state in-process for development.",
             metadata={"source": "lg_memory",  "page": 5}),
    Document(page_content="PostgreSQL checkpointer persists state across restarts.",
             metadata={"source": "lg_postgres","page": 6}),
    Document(page_content="RAG grounds LLM answers in retrieved external documents.",
             metadata={"source": "rag_intro",  "page": 1}),
    Document(page_content="FAISS is a fast in-process similarity search library.",
             metadata={"source": "rag_faiss",  "page": 2}),
]

embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore  = FAISS.from_documents(docs, embeddings)


# ── 1. Default retriever (top-4 similarity) ───────────────────────────────────

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print("=== Default retriever (k=4) ===")
results = retriever.invoke("How does LangGraph manage state?")
for doc in results:
    print(f"  source={doc.metadata['source']:<15}  {doc.page_content[:70]}")


# ── 2. MMR retriever — diverse results ───────────────────────────────────────
# MMR balances relevance with diversity: avoids returning near-duplicate chunks.
# Useful when the corpus has many similar passages and you want coverage.

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 8},   # fetch_k candidates, return k diverse ones
)

print("\n=== MMR retriever (diverse) ===")
results_mmr = mmr_retriever.invoke("LangGraph state and persistence")
for doc in results_mmr:
    print(f"  source={doc.metadata['source']:<15}  {doc.page_content[:70]}")


# ── 3. Score threshold — only high-confidence docs ───────────────────────────

threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 6},
)

print("\n=== Score threshold (≥0.5) ===")
results_thresh = threshold_retriever.invoke("Kubernetes deployment scaling")
print(f"  Returned {len(results_thresh)} docs (low relevance query — few above threshold)")


# ── 4. Retriever is a Runnable — use in LCEL chain ───────────────────────────

print("\n=== Retriever as Runnable ===")
# .batch() retrieves for multiple queries at once
queries = ["nodes in LangGraph", "vector store options"]
batched = retriever.batch(queries)
for q, docs_list in zip(queries, batched):
    print(f"\n  Query: {q!r}")
    for d in docs_list[:2]:
        print(f"    → {d.metadata['source']}: {d.page_content[:60]}")
