# Interview Questions — 09 RAG

---

## Q1. What is RAG and why is it better than fine-tuning for dynamic knowledge?

**Answer:**
RAG (Retrieval-Augmented Generation) grounds LLM outputs in external documents retrieved at query time. Instead of baking knowledge into model weights, you retrieve the most relevant passages and pass them in the context window.

| | RAG | Fine-tuning |
|--|-----|-------------|
| Update knowledge | Add docs to vector store — no retraining | Retrain the model |
| Cost | Cheap (embedding + retrieval) | Expensive (GPU hours) |
| Freshness | Real-time | Snapshot at training time |
| Citations | Built-in (metadata) | Difficult |
| Hallucination | Reduced (grounded in docs) | Still present |

Fine-tuning is better for teaching the model *how* to behave (style, format, task type). RAG is better for keeping *what* it knows current.

---

## Q2. What is a `Document` in LangChain and what are its fields?

**Answer:**
`Document` is LangChain's unit of text. It has two fields:

- `page_content` — the raw text the embedding model and LLM will read.
- `metadata` — an arbitrary dict (source filename, page number, URL, author, timestamp). Metadata is preserved through splitting, embedding, and retrieval — it is the foundation of citations.

```python
Document(
    page_content="LangGraph builds stateful agent workflows.",
    metadata={"source": "overview.txt", "page": 1}
)
```

Every loader (`TextLoader`, `PyPDFLoader`, `WebBaseLoader`) produces a list of `Document` objects.

---

## Q3. Why split documents into chunks? What are `chunk_size` and `chunk_overlap`?

**Answer:**
Embedding models have token limits (~8k for OpenAI). A 100-page PDF must be split before embedding. Smaller chunks also improve retrieval precision — retrieving a focused 300-character paragraph is more useful than a 10-page chapter.

- `chunk_size` — maximum characters (or tokens) per chunk.
- `chunk_overlap` — characters repeated between adjacent chunks. Prevents key sentences at boundaries from being cut and becoming unretrievable.

`RecursiveCharacterTextSplitter` is the recommended default — it tries to split on paragraph breaks, then sentences, then words, keeping chunks coherent.

---

## Q4. What is an embedding and what does it mean for two texts to be "close"?

**Answer:**
An embedding is a dense vector of floats produced by a model (e.g. `text-embedding-3-small` → 1536 dimensions). The model is trained so that semantically similar texts produce vectors that are geometrically close (low cosine distance or L2 distance).

"What is LangGraph?" and "LangGraph is a graph-based agent framework" produce nearby vectors even though they share few words — the model captures *meaning*, not keywords. This enables semantic search: find documents about a concept even when the exact words don't match.

---

## Q5. What is FAISS and when should you replace it in production?

**Answer:**
FAISS (Facebook AI Similarity Search) is an in-process library that stores embedding vectors in RAM and performs fast nearest-neighbour search. It is ideal for:
- Development and prototyping
- Small-to-medium corpora (up to a few million vectors on a large machine)
- Single-process applications

**Replace with a managed vector DB when you need:**
- Persistence across restarts (FAISS must be saved/reloaded from disk)
- Horizontal scaling across multiple replicas
- Real-time updates (insert/delete while serving queries)
- Metadata filtering at query time
- Managed infrastructure (backups, monitoring)

Options: **pgvector** (Postgres extension — simple if you already use Postgres), **Pinecone** (hosted, no ops), **Weaviate** (open source, feature-rich).

---

## Q6. What does `RunnableParallel` do in the RAG chain?

**Answer:**
`RunnableParallel` runs multiple branches on the same input simultaneously and collects their results into a dict:

```python
RunnableParallel({
    "context":  retriever | format_docs,   # retrieves & formats docs
    "question": RunnablePassthrough(),      # passes the question through unchanged
})
```

Both branches receive the user's question string. The dict output `{"context": "...", "question": "..."}` then fills the prompt template. Without `RunnableParallel` you'd have to manually thread the question string through two separate calls.

---

## Q7. Why split `retrieve_node` and `generate_node` into separate LangGraph nodes?

**Answer:**
Two practical reasons:

1. **Observability** — in LangSmith, each node appears as a separate span. You can see exactly what was retrieved and what was generated independently, which makes debugging much easier than a single opaque chain.

2. **Composability** — downstream nodes can read `state["context"]` without re-retrieving. For example, a `citation_node` that formats sources can run after `generate_node` without making another embedding query. A `guardrail_node` can inspect retrieved docs before generation.

A single chain is fine for simple use cases; separate nodes are the right choice when the graph has more than two steps.

---

## Q8. What is Maximal Marginal Relevance (MMR) and when should you use it?

**Answer:**
MMR balances two goals when selecting k documents:
1. **Relevance** — each selected document should be semantically close to the query.
2. **Diversity** — selected documents should not be near-duplicates of each other.

Without MMR, if the corpus contains five nearly-identical paragraphs about the same topic, the top-k retrieval may return all five — wasting context window space. MMR avoids this by penalising candidates that are too similar to already-selected documents.

Use MMR when:
- Your corpus has many near-duplicate chunks (e.g. repeated boilerplate in docs).
- You want broad coverage of a topic, not five variations of the same sentence.
- Context window budget is tight and you need maximum information density.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20},
)
```
