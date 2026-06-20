"""
Concept: Embeddings + vector store — turning text into searchable vectors.

Embeddings convert text into dense numeric vectors (lists of floats).
Semantically similar texts produce vectors that are close together in
high-dimensional space — this is what makes similarity search possible.

FAISS (Facebook AI Similarity Search) stores these vectors in memory
and finds the nearest neighbours of a query vector in milliseconds.

Pipeline:
  chunks → OpenAIEmbeddings → float vectors → FAISS index

Run:
    python 09_rag/03_embeddings_vectorstore.py
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# ── Sample corpus ─────────────────────────────────────────────────────────────

docs = [
    Document(page_content="LangGraph models agent logic as a directed graph with nodes and edges.",
             metadata={"source": "langgraph_intro", "topic": "langgraph"}),
    Document(page_content="StateGraph is the core LangGraph primitive. Define state as a TypedDict.",
             metadata={"source": "langgraph_state",  "topic": "langgraph"}),
    Document(page_content="InMemorySaver is a checkpointer that stores state in a Python dict for development.",
             metadata={"source": "checkpointing",    "topic": "persistence"}),
    Document(page_content="AsyncPostgresSaver persists LangGraph state to PostgreSQL for production use.",
             metadata={"source": "checkpointing",    "topic": "persistence"}),
    Document(page_content="RAG retrieves relevant documents from a vector store before calling the LLM.",
             metadata={"source": "rag_intro",        "topic": "rag"}),
    Document(page_content="FAISS is an efficient in-process vector store suitable for small-to-medium corpora.",
             metadata={"source": "vector_stores",    "topic": "rag"}),
    Document(page_content="OpenAI's text-embedding-3-small model produces 1536-dimensional vectors.",
             metadata={"source": "embeddings",       "topic": "rag"}),
    Document(page_content="Chunk overlap ensures sentences at chunk boundaries are fully represented.",
             metadata={"source": "splitting",        "topic": "rag"}),
]


# ── 1. Create embeddings model ────────────────────────────────────────────────
# text-embedding-3-small: cheap, fast, 1536 dimensions, great for RAG.

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ── 2. Embed a single string (for inspection) ─────────────────────────────────

sample_vector = embeddings.embed_query("What is LangGraph?")
print("=== Single embedding ===")
print(f"Dimensions : {len(sample_vector)}")
print(f"First 5    : {[round(v, 4) for v in sample_vector[:5]]}")


# ── 3. Build FAISS vector store from documents ────────────────────────────────
# from_documents embeds all page_content strings and indexes them.

print("\n=== Building FAISS index ===")
vectorstore = FAISS.from_documents(docs, embeddings)
print(f"Indexed {vectorstore.index.ntotal} vectors")


# ── 4. Similarity search ──────────────────────────────────────────────────────

print("\n=== Similarity search: 'How does LangGraph store state?' ===")
results = vectorstore.similarity_search(
    "How does LangGraph store state?",
    k=3,   # return top-3 most similar chunks
)
for i, doc in enumerate(results):
    print(f"\n[{i}] score source={doc.metadata['source']}")
    print(f"     {doc.page_content}")


# ── 5. Search with relevance scores ──────────────────────────────────────────
# Returns (Document, score) tuples — lower L2 distance = more similar.

print("\n=== Search with scores ===")
results_with_scores = vectorstore.similarity_search_with_score(
    "vector store for production",
    k=3,
)
for doc, score in results_with_scores:
    print(f"  score={score:.4f}  source={doc.metadata['source']}")
    print(f"  {doc.page_content[:80]}")


# ── 6. Save and reload the index ──────────────────────────────────────────────
# FAISS can be serialised to disk — avoids re-embedding on every restart.

import tempfile, os

with tempfile.TemporaryDirectory() as tmpdir:
    vectorstore.save_local(tmpdir)
    print(f"\n=== Saved index to {tmpdir} ===")
    print("Files:", os.listdir(tmpdir))

    reloaded = FAISS.load_local(
        tmpdir,
        embeddings,
        allow_dangerous_deserialization=True,   # required for pickle-based load
    )
    print(f"Reloaded index has {reloaded.index.ntotal} vectors")
