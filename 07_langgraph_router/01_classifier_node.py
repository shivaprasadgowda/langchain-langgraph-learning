"""
Concept: Classifier node — structured output inside a LangGraph node.

This combines two earlier patterns:
  - Section 03: with_structured_output for typed classification
  - Section 06: a node is (state) -> dict

The classifier node reads the latest human message from state,
classifies its intent, and writes the result back to state.
Downstream conditional edges read state["intent"] to decide routing.

Run:
    python 07_langgraph_router/01_classifier_node.py
"""

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages:   Annotated[list[BaseMessage], add_messages]
    intent:     str   # set by classifier_node, read by router and specialist nodes
    confidence: str   # informational — useful for logging / observability


# ── Structured output schema (from section 03 pattern) ───────────────────────

class Classification(BaseModel):
    """Intent classification for a DevOps support chatbot."""

    intent: Literal["jira", "aws", "kubernetes", "general"] = Field(
        description=(
            "jira        — Jira tickets, sprints, epics, or project management.\n"
            "aws         — AWS services, EC2, S3, Lambda, IAM, costs, or deployments.\n"
            "kubernetes  — pods, deployments, services, Helm charts, or kubectl.\n"
            "general     — everything else."
        )
    )
    confidence: Literal["high", "medium", "low"]


# ── Classifier node ───────────────────────────────────────────────────────────

_classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_classifier_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an intent classifier for a DevOps support chatbot. "
        "Classify the user message into one of the defined intents.",
    ),
    ("human", "{message}"),
])
_classifier_chain = _classifier_prompt | _classifier_llm.with_structured_output(Classification)


def classifier_node(state: State) -> dict:
    """Classify the intent of the most recent human message."""
    # Get the last human message from state
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"), None
    )
    if last_human is None:
        return {"intent": "general", "confidence": "low"}

    result: Classification = _classifier_chain.invoke({"message": last_human.content})

    print(f"  [classifier] intent={result.intent!r}  confidence={result.confidence!r}")
    return {"intent": result.intent, "confidence": result.confidence}


# ── Minimal graph to demonstrate the classifier node alone ────────────────────

builder = StateGraph(State)
builder.add_node("classifier", classifier_node)
builder.add_edge(START, "classifier")
builder.add_edge("classifier", END)

app = builder.compile()


# ── Test ──────────────────────────────────────────────────────────────────────

test_messages = [
    "Create a Jira ticket for the login 500 error.",
    "Why is our S3 cost up 40% this month?",
    "The auth-service pod is in CrashLoopBackOff.",
    "What does SOLID stand for?",
]

print(f"{'Message':<55} {'Intent':<12} {'Confidence'}")
print("-" * 80)

for text in test_messages:
    result = app.invoke({
        "messages": [HumanMessage(content=text)],
        "intent": "",
        "confidence": "",
    })
    print(f"{text[:53]:<55} {result['intent']:<12} {result['confidence']}")
