"""
Concept: Citations — returning source metadata alongside the answer.

A plain RAG chain returns only the generated text. In production you
usually also need:
  - Which documents were retrieved (for "show your sources")
  - Which specific passages were used (for inline citations)
  - Confidence / relevance scores (for filtering low-quality answers)

Two complementary approaches are shown here:

  Approach A — source list: return the metadata of every retrieved doc.
               Easy, no extra LLM call needed.

  Approach B — structured citations: ask the model to identify which
               sources it actually used and quote the key passage.
               Requires a second structured-output call but is more precise.

Run:
    python 09_rag/07_citations_concept.py
"""

from dotenv import load_dotenv
from typing import Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

load_dotenv()


# ── Knowledge base ────────────────────────────────────────────────────────────

docs = [
    Document(page_content="LangGraph builds stateful agent workflows as directed graphs.",
             metadata={"source": "langgraph_overview.txt", "page": 1, "author": "LangChain Team"}),
    Document(page_content="StateGraph defines the state schema as a TypedDict shared by all nodes.",
             metadata={"source": "langgraph_overview.txt", "page": 2, "author": "LangChain Team"}),
    Document(page_content="MemorySaver stores state in a dict. AsyncPostgresSaver is for production.",
             metadata={"source": "persistence_guide.txt",  "page": 1, "author": "LangChain Team"}),
    Document(page_content="thread_id in config isolates separate conversations within one graph.",
             metadata={"source": "persistence_guide.txt",  "page": 2, "author": "LangChain Team"}),
    Document(page_content="RAG retrieves relevant documents before calling the LLM to ground the answer.",
             metadata={"source": "rag_overview.txt",       "page": 1, "author": "LangChain Team"}),
    Document(page_content="FAISS is an efficient in-process vector store for small-to-medium corpora.",
             metadata={"source": "rag_overview.txt",       "page": 2, "author": "LangChain Team"}),
]

embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
llm         = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH A: Source list — metadata from every retrieved doc
# ═══════════════════════════════════════════════════════════════════════════════

def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context provided.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

# Chain returns both the answer and the retrieved docs
chain_with_sources = RunnableParallel({
    "answer":   (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | rag_prompt | llm | StrOutputParser()
    ),
    "source_docs": retriever,   # raw Document objects
})

print("=" * 60)
print("APPROACH A: Source list")
print("=" * 60)

result = chain_with_sources.invoke("What checkpointer should I use in production?")
print(f"Answer: {result['answer']}")
print("\nSources:")
for doc in result["source_docs"]:
    print(f"  {doc.metadata['source']}  (page {doc.metadata['page']})")
    print(f"    {doc.page_content[:70]}")


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH B: Structured citations — model identifies which sources it used
# ═══════════════════════════════════════════════════════════════════════════════

class Citation(BaseModel):
    source: str = Field(description="The source filename of the cited document.")
    page:   int = Field(description="The page number of the cited document.")
    quote:  str = Field(description="The exact short passage from the document that supports the answer.")


class AnswerWithCitations(BaseModel):
    """An answer grounded in specific citations from the retrieved documents."""
    answer:    str            = Field(description="The answer to the user's question.")
    citations: list[Citation] = Field(description="List of sources that directly support the answer.")
    confidence: str           = Field(description="'high', 'medium', or 'low' based on how well the context answers the question.")


citation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the question using the provided context. "
        "For each claim in your answer, cite the exact source and quote the passage. "
        "Only cite passages you actually used.\n\n"
        "Context (with source labels):\n{context}",
    ),
    ("human", "{question}"),
])

def format_docs_with_labels(docs: list[Document]) -> str:
    """Include source metadata in the formatted context so the model can cite it."""
    parts = []
    for doc in docs:
        label = f"[{doc.metadata['source']} p.{doc.metadata['page']}]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

structured_llm = llm.with_structured_output(AnswerWithCitations)

def rag_with_citations(question: str) -> AnswerWithCitations:
    retrieved = retriever.invoke(question)
    context   = format_docs_with_labels(retrieved)
    return structured_llm.invoke(
        citation_prompt.format_messages(context=context, question=question)
    )


print("\n" + "=" * 60)
print("APPROACH B: Structured citations")
print("=" * 60)

result2 = rag_with_citations("How does LangGraph isolate conversations?")
print(f"Answer    : {result2.answer}")
print(f"Confidence: {result2.confidence}")
print("Citations:")
for c in result2.citations:
    print(f"  [{c.source} p.{c.page}] \"{c.quote[:70]}\"")
