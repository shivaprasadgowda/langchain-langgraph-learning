"""
Concept: RAG as a LangGraph node.

Wrapping the RAG chain inside a LangGraph node enables:
  - Combining retrieval with other nodes (classifier, tools, human approval)
  - Persisting retrieved context in state for downstream nodes
  - Routing: only call RAG for certain intents, not all queries

Graph shape:
    START → retrieve_node → generate_node → END

  retrieve_node  — fetches relevant chunks and stores them in state
  generate_node  — reads state["context"] and state["messages"], generates answer

Splitting retrieve and generate into separate nodes makes each independently
testable and observable in LangSmith traces.

Run:
    python 09_rag/06_rag_langgraph_node.py
"""

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


# ── Knowledge base ────────────────────────────────────────────────────────────

_docs = [
    Document(page_content="LangGraph builds stateful agent workflows as directed graphs with nodes and edges.",
             metadata={"source": "overview.txt",    "page": 1}),
    Document(page_content="StateGraph is the core primitive. State is a TypedDict all nodes read and write.",
             metadata={"source": "overview.txt",    "page": 2}),
    Document(page_content="Nodes are plain Python functions: (state) -> dict (partial update).",
             metadata={"source": "nodes.txt",       "page": 1}),
    Document(page_content="Conditional edges call a routing function on state to choose the next node.",
             metadata={"source": "edges.txt",       "page": 1}),
    Document(page_content="MemorySaver is for development; AsyncPostgresSaver is for production persistence.",
             metadata={"source": "persistence.txt", "page": 1}),
    Document(page_content="thread_id in config isolates conversations within the same compiled graph.",
             metadata={"source": "persistence.txt", "page": 2}),
]

_embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
_vectorstore = FAISS.from_documents(_docs, _embeddings)
_retriever   = _vectorstore.as_retriever(search_kwargs={"k": 3})


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context:  list[Document]   # populated by retrieve_node, read by generate_node


# ── Nodes ─────────────────────────────────────────────────────────────────────

def retrieve_node(state: State) -> dict:
    """Retrieve relevant documents for the latest human question."""
    question = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"),
        "",
    )
    docs = _retriever.invoke(question)
    print(f"  [retrieve] fetched {len(docs)} chunks for: {question[:50]!r}")
    return {"context": docs}


_rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer using ONLY the provided context. "
        "If the context lacks the answer, say so.\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def generate_node(state: State) -> dict:
    """Generate an answer from retrieved context."""
    question = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"),
        "",
    )
    context_text = "\n\n---\n\n".join(d.page_content for d in state["context"])
    response = _llm.invoke(
        _rag_prompt.format_messages(context=context_text, question=question)
    )
    print(f"  [generate] answer: {response.content[:80]}...")
    return {"messages": [response]}


# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(State)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

builder.add_edge(START,      "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

app = builder.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

questions = [
    "What is a StateGraph?",
    "Which checkpointer should I use for production?",
    "What is thread_id used for?",
]

for q in questions:
    print(f"\n{'─'*60}")
    print(f"User: {q}")
    result = app.invoke({
        "messages": [HumanMessage(content=q)],
        "context":  [],
    })
    print(f"Answer: {result['messages'][-1].content}")
    print(f"Sources: {[d.metadata['source'] for d in result['context']]}")
