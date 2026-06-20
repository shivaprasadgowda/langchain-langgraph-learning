# RAG Interview Questions

---

## Q1. What is RAG and what problem does it solve?

**Answer:**
RAG (Retrieval-Augmented Generation) gives an LLM access to knowledge it wasn't trained on — private documents, recent events, internal databases — without fine-tuning.

**The problem:** LLMs have a knowledge cutoff date and no access to private data. Putting all documents in the context window is too expensive and hits token limits.

**The solution:**
1. **Index** — embed documents into vectors and store in a vector database
2. **Retrieve** — for each query, embed it and find semantically similar document chunks
3. **Generate** — pass the retrieved chunks + query to the LLM as context

```
User query → embed → vector search → top-K chunks
                                          ↓
                                    [CONTEXT] + [QUERY] → LLM → answer
```

RAG is more cost-effective than fine-tuning (no GPU training), always shows the source documents (traceable), and can be updated by re-indexing (unlike fine-tuning which bakes knowledge in).

---

## Q2. Walk through the indexing pipeline.

**Answer:**
```
Raw documents (PDF, HTML, text)
        ↓
  Document loading (TextLoader, PyPDFLoader, WebBaseLoader)
        ↓
  Text splitting (RecursiveCharacterTextSplitter)
    chunk_size=1000, chunk_overlap=200
        ↓
  Embedding (OpenAIEmbeddings("text-embedding-3-small"))
    Each chunk → 1536-dim float vector
        ↓
  Vector store (FAISS, pgvector, Chroma, Pinecone)
    Store (vector, chunk_text, metadata)
```

Key decisions:
- **Chunk size** — too small: context lost. Too large: irrelevant text dilutes the answer and wastes tokens. 512-1000 chars is typical.
- **Chunk overlap** — prevents splitting a sentence at a chunk boundary. 10-20% of chunk_size is typical.
- **Embedding model** — `text-embedding-3-small` is fast and cheap. `text-embedding-3-large` is more accurate but 5× the cost.
- **Metadata** — always store source, page number, section, date. Used for filtering and citations.

---

## Q3. What is the difference between similarity search, MMR, and threshold search?

**Answer:**

| Type | How it works | Use case |
|------|-------------|---------|
| **Similarity search** (k=N) | Returns top-N chunks by cosine similarity | General RAG; when you want the most relevant chunks |
| **MMR** (Maximal Marginal Relevance) | Balances relevance AND diversity; avoids returning N nearly-identical chunks | When the vector store has many similar docs and you want variety |
| **Similarity with threshold** | Returns only chunks with similarity score ≥ threshold | When you want to return nothing rather than an irrelevant answer |

```python
# Similarity — top 4 by cosine distance
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# MMR — 4 diverse results, fetches 8 candidates first
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 8},
)

# Threshold — only return if similarity ≥ 0.75
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.75},
)
```

MMR is important when users ask broad questions about a topic where many similar chunks exist — without MMR, all 4 results might cover the same subtopic.

---

## Q4. Write the LCEL RAG chain pattern.

**Answer:**
```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context below. Say 'I don't know' if the answer isn't in the context.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    RunnableParallel({
        "context":  retriever | format_docs,
        "question": RunnablePassthrough(),
    })
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What causes Kubernetes pod restarts?")
```

The `RunnableParallel` runs retrieval and passthrough simultaneously. The `|` pipe feeds both into the prompt, which feeds into the LLM, which feeds into the parser.

---

## Q5. How do you add citations to RAG output?

**Answer:**
Two approaches:

**Approach A — parallel chain (returns source docs alongside answer):**
```python
chain_with_sources = RunnableParallel({
    "answer":      rag_chain,
    "source_docs": retriever,
})
result = chain_with_sources.invoke("What is a pod?")
# result["answer"] — the text answer
# result["source_docs"] — list of Document objects with metadata
```

**Approach B — structured output with inline citations:**
```python
class Citation(BaseModel):
    source_id: str
    quote:     str

class AnswerWithCitations(BaseModel):
    answer:    str
    citations: list[Citation]

def format_docs_with_labels(docs):
    return "\n\n".join(
        f"[{i+1}] {d.page_content}\nSource: {d.metadata.get('source', '?')}"
        for i, d in enumerate(docs)
    )

# Prompt tells model to cite using [1], [2] notation
structured_llm = llm.with_structured_output(AnswerWithCitations)
```

Approach A is simpler; Approach B enables inline `[1]` footnotes in the UI.

---

## Q6. What are the main failure modes of RAG and how do you address each?

**Answer:**

| Failure | Cause | Fix |
|---------|-------|-----|
| **Wrong chunks retrieved** | Query phrasing doesn't match chunk vocabulary | Rewrite query (HyDE, multi-query), better chunking strategy |
| **Answer contradicts retrieved docs** | LLM hallucinating despite context | Stricter prompt ("only use context provided"), reduce temperature |
| **Missing information** | Relevant chunk not in top-K | Increase K, use MMR for diversity, re-rank results |
| **Context too long** | Too many/large chunks → token limit | Reduce K or chunk size, use long-context model, summarize chunks |
| **Stale answers** | Vector store not re-indexed after docs change | Trigger re-indexing on document update; store `updated_at` metadata |
| **Low similarity threshold blocks valid results** | Threshold too high | Tune threshold on eval set, use adaptive threshold |

**HyDE (Hypothetical Document Embedding):** generate a hypothetical answer and embed that, then retrieve. The hypothetical answer is closer in embedding space to the actual document than the original question is.

---

## Q7. How does a RAG node fit into a LangGraph agent?

**Answer:**
```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context:  str    # retrieved chunks stored here

def retrieve_node(state: State) -> dict:
    query    = state["messages"][-1].content
    docs     = retriever.invoke(query)
    context  = "\n\n".join(d.page_content for d in docs)
    return {"context": context}   # stored in state for next node

def generate_node(state: State) -> dict:
    system = SystemMessage(content=f"Use this context:\n\n{state['context']}")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

builder.add_edge(START,     "retrieve")
builder.add_edge("retrieve","generate")
builder.add_edge("generate", END)
```

Benefits of LangGraph RAG over LCEL RAG:
- Retrieval result is in state → visible in LangSmith trace per node
- Can add a `rerank_node` between retrieve and generate
- Can add a `grade_docs_node` that routes to retrieve again if chunks are irrelevant (self-corrective RAG)

---

## Q8. When would you use pgvector vs Pinecone vs FAISS?

**Answer:**

| Store | When to use |
|-------|-------------|
| **FAISS** | Local dev, prototyping, < 1M vectors, no infra budget |
| **pgvector** | You already use PostgreSQL; < 10M vectors; want JOINs with relational data; simpler infra |
| **Chroma** | Local/self-hosted, easy setup, good for < 5M vectors |
| **Pinecone** | > 10M vectors; need managed scaling, namespacing, and metadata filtering at scale; willing to pay SaaS cost |
| **Weaviate / Qdrant** | Need hybrid search (keyword + vector), open-source, self-hosted at scale |

**Decision rule:**
- Start with FAISS for prototyping.
- Move to pgvector if you're already on PostgreSQL — avoid a second DB.
- Move to Pinecone/Weaviate when you need millions of vectors, managed scaling, or advanced metadata filtering that pgvector's `ivfflat` index can't handle efficiently.
