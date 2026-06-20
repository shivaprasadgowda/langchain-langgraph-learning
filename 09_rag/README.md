# 09 — RAG (Retrieval-Augmented Generation)

> Status: **Complete**

## What This Section Covers

- Document loading
- Text splitting
- Embeddings and vector store (FAISS)
- Retriever
- RAG chain
- RAG node inside a LangGraph graph
- Citations concept

## Files

| File | Purpose |
|------|---------|
| `01_document_loading.py` | `Document` structure, inline corpus, loader reference — no API key needed |
| `02_text_splitting.py` | `RecursiveCharacterTextSplitter`, chunk_size/overlap effects, boundary demo — no API key needed |
| `03_embeddings_vectorstore.py` | `OpenAIEmbeddings`, `FAISS.from_documents`, similarity search with scores, save/reload |
| `04_retriever.py` | `as_retriever`, similarity vs MMR vs score-threshold, `.batch()` as Runnable |
| `05_rag_chain.py` | Full `RunnableParallel` RAG chain; out-of-context query handled gracefully |
| `06_rag_langgraph_node.py` | `retrieve_node` + `generate_node` in a two-step graph; context in state |
| `07_citations_concept.py` | Approach A (source list via `RunnableParallel`) + Approach B (structured citations via `with_structured_output`) |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
python 09_rag/01_document_loading.py        # no API key
python 09_rag/02_text_splitting.py          # no API key
python 09_rag/03_embeddings_vectorstore.py
python 09_rag/04_retriever.py
python 09_rag/05_rag_chain.py
python 09_rag/06_rag_langgraph_node.py
python 09_rag/07_citations_concept.py
```

## RAG Pipeline

```
Documents
   │
   ▼
RecursiveCharacterTextSplitter  (chunk_size, chunk_overlap)
   │
   ▼
OpenAIEmbeddings → float vectors
   │
   ▼
FAISS vector store
   │
   ▼  (query time)
Retriever.invoke(question) → top-k Documents
   │
   ├── format_docs ──► context string ──► prompt | llm | parser → answer
   └── metadata ──────────────────────────────────────────────→ citations
```

## Key Concepts

**`Document`** — `page_content` + `metadata`; metadata survives splitting and retrieval; foundation of citations.

**`RecursiveCharacterTextSplitter`** — splits on `\n\n` → `\n` → ` ` → char; overlap prevents boundary loss.

**`FAISS.from_documents`** — embeds all chunks and builds an index in one call; `.save_local()` / `.load_local()` for persistence.

**`as_retriever()`** — converts vectorstore to a `Runnable`; supports `similarity`, `mmr`, `similarity_score_threshold`.

**`RunnableParallel`** — runs retrieval and passthrough simultaneously; feeds both into the prompt.

**Separate retrieve/generate nodes** — better observability in LangSmith; downstream nodes can read `state["context"]` without re-fetching.
