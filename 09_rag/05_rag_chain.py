"""
Concept: Full RAG LCEL chain — retrieve → format → generate.

Chain shape:
    question
       │
       ├──► retriever ──► format_docs ──┐
       │                                ▼
       └────────────────────────► prompt | llm | StrOutputParser
                                         │
                                       answer

The parallel branch (RunnableParallel) passes both the retrieved context
and the original question to the prompt template simultaneously.

Run:
    python 09_rag/05_rag_chain.py
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# ── 1. Build the knowledge base ───────────────────────────────────────────────

docs = [
    Document(page_content="LangGraph is a library for building stateful agent workflows as directed graphs.",
             metadata={"source": "overview.txt", "page": 1}),
    Document(page_content="StateGraph is the core LangGraph primitive. State is a TypedDict shared by all nodes.",
             metadata={"source": "overview.txt", "page": 2}),
    Document(page_content="Nodes are Python functions with signature (state: State) -> dict.",
             metadata={"source": "nodes.txt",    "page": 1}),
    Document(page_content="Conditional edges route execution by calling a function on state that returns a node name.",
             metadata={"source": "edges.txt",    "page": 1}),
    Document(page_content="MemorySaver is an in-process checkpointer for development. Use AsyncPostgresSaver in production.",
             metadata={"source": "persistence.txt", "page": 1}),
    Document(page_content="thread_id in config isolates separate conversations within the same graph.",
             metadata={"source": "persistence.txt", "page": 2}),
    Document(page_content="RAG stands for Retrieval-Augmented Generation. It grounds LLM output in retrieved documents.",
             metadata={"source": "rag.txt",      "page": 1}),
    Document(page_content="FAISS is an efficient in-memory vector store. For production scale use Pinecone or pgvector.",
             metadata={"source": "rag.txt",      "page": 2}),
]

embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})


# ── 2. Format retrieved docs for the prompt ───────────────────────────────────

def format_docs(docs: list[Document]) -> str:
    """Concatenate document contents with a separator."""
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ── 3. RAG prompt template ────────────────────────────────────────────────────

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question using ONLY the "
        "provided context. If the context does not contain the answer, say "
        "'I don't have enough information in the provided context.'\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 4. Assemble the RAG chain ─────────────────────────────────────────────────
# RunnableParallel runs both branches for the same input simultaneously:
#   context  branch: question → retriever → format_docs
#   question branch: question → passthrough (unchanged)
# Both outputs feed into the prompt template.

rag_chain = (
    RunnableParallel({
        "context":  retriever | format_docs,
        "question": RunnablePassthrough(),
    })
    | prompt
    | llm
    | StrOutputParser()
)


# ── 5. Run queries ────────────────────────────────────────────────────────────

questions = [
    "What is a StateGraph in LangGraph?",
    "What checkpointer should I use in production?",
    "How are conditional edges different from normal edges?",
    "What is the capital of France?",   # out-of-context question
]

for q in questions:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}")


# ── 6. Inspect retrieved context for one query ───────────────────────────────

print("\n=== Retrieved context for 'What is thread_id?' ===")
context_docs = retriever.invoke("What is thread_id?")
for i, doc in enumerate(context_docs):
    print(f"\n[chunk {i}] source={doc.metadata['source']}")
    print(f"  {doc.page_content}")
