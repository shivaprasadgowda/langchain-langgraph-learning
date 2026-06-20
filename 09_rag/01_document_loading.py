"""
Concept: Document loading — the first step of any RAG pipeline.

A LangChain Document has two fields:
  page_content — the raw text the model will read
  metadata     — arbitrary dict: source, page number, author, URL, etc.
                 Metadata is what enables citations (section 07).

In production you load documents from files, databases, APIs, or web pages.
Here we create them inline so the file runs without any external dependencies.

Run:
    python 09_rag/01_document_loading.py
"""

from langchain_core.documents import Document


# ── 1. Create Documents manually ──────────────────────────────────────────────
# This is what every loader (TextLoader, PyPDFLoader, WebBaseLoader) produces.

docs = [
    Document(
        page_content=(
            "LangGraph is a library for building stateful, multi-actor applications "
            "with LLMs. It extends LangChain by modelling agent logic as a directed "
            "graph where nodes are computation steps and edges are control flow."
        ),
        metadata={"source": "docs/langgraph_overview.txt", "section": "intro"},
    ),
    Document(
        page_content=(
            "A StateGraph is the core LangGraph primitive. You define a TypedDict "
            "as the state schema, add nodes (Python functions), add edges, and "
            "compile the graph to get a runnable application."
        ),
        metadata={"source": "docs/langgraph_overview.txt", "section": "stategraph"},
    ),
    Document(
        page_content=(
            "LangGraph checkpointers persist graph state between invocations. "
            "InMemorySaver is used for development. AsyncPostgresSaver is recommended "
            "for production because it survives process restarts and supports "
            "horizontal scaling."
        ),
        metadata={"source": "docs/langgraph_checkpointing.txt", "section": "checkpointers"},
    ),
    Document(
        page_content=(
            "RAG (Retrieval-Augmented Generation) grounds LLM answers in external "
            "knowledge. The pipeline: load documents → split into chunks → embed → "
            "store in a vector store → retrieve relevant chunks → pass to LLM."
        ),
        metadata={"source": "docs/rag_overview.txt", "section": "intro"},
    ),
    Document(
        page_content=(
            "FAISS (Facebook AI Similarity Search) is an efficient in-process vector "
            "store. It is ideal for development and small-to-medium document sets. "
            "For production at scale, consider Pinecone, Weaviate, or pgvector."
        ),
        metadata={"source": "docs/vector_stores.txt", "section": "faiss"},
    ),
]


# ── 2. Inspect the Document structure ─────────────────────────────────────────

print("=== Document structure ===")
doc = docs[0]
print(f"type         : {type(doc).__name__}")
print(f"page_content : {doc.page_content[:80]}...")
print(f"metadata     : {doc.metadata}")


# ── 3. Show all documents ─────────────────────────────────────────────────────

print(f"\n=== All {len(docs)} documents ===")
for i, d in enumerate(docs):
    print(f"\n[{i}] source  : {d.metadata['source']}")
    print(f"    section : {d.metadata['section']}")
    print(f"    length  : {len(d.page_content)} chars")
    print(f"    preview : {d.page_content[:70]}...")


# ── 4. What loaders exist (reference — no import needed) ─────────────────────

print("\n=== Common loaders (reference) ===")
loaders = [
    ("TextLoader",        "langchain_community.document_loaders", "Plain .txt files"),
    ("PyPDFLoader",       "langchain_community.document_loaders", "PDF files"),
    ("WebBaseLoader",     "langchain_community.document_loaders", "Web pages via URL"),
    ("DirectoryLoader",   "langchain_community.document_loaders", "All files in a folder"),
    ("NotionDBLoader",    "langchain_community.document_loaders", "Notion database pages"),
    ("ConfluenceLoader",  "langchain_community.document_loaders", "Confluence wiki pages"),
]
for name, module, desc in loaders:
    print(f"  {name:<25} — {desc}")
