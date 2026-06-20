"""
Concept: Intent classification with structured output.

Classification is one of the highest-value uses of structured output:
  - The model picks from a fixed set of labels (Literal[...]).
  - temperature=0 makes the choice deterministic.
  - The result is a typed Python object you can branch on directly.

This pattern is the foundation of the LangGraph router in section 07.

Run:
    python 03_structured_output/03_classification.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


# ── Schema ────────────────────────────────────────────────────────────────────

class IntentClassification(BaseModel):
    """Classify the intent of a user message for a DevOps support chatbot."""

    intent: Literal["jira", "aws", "kubernetes", "general"] = Field(
        description=(
            "jira        — creating/updating Jira tickets or sprints.\n"
            "aws         — AWS services, costs, deployments, or permissions.\n"
            "kubernetes  — pods, deployments, services, namespaces, or kubectl.\n"
            "general     — anything else."
        )
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How confident the classification is."
    )
    reasoning: str = Field(
        description="One sentence explaining why this intent was chosen."
    )


# ── Chain ─────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an intent classifier for a DevOps support chatbot. "
        "Classify user messages into one of the defined intents.",
    ),
    ("human", "{user_message}"),
])

chain = template | llm.with_structured_output(IntentClassification)


# ── Test cases ────────────────────────────────────────────────────────────────

test_messages = [
    "Create a Jira ticket for the login bug found in QA.",
    "Why is our EC2 bill so high this month?",
    "The payment-service pod keeps crashing with OOMKilled.",
    "Can you explain what a microservice is?",
]

print(f"{'Message':<55} {'Intent':<12} {'Confidence'}")
print("-" * 85)

for msg in test_messages:
    result: IntentClassification = chain.invoke({"user_message": msg})
    print(f"{msg[:53]:<55} {result.intent:<12} {result.confidence}")

# ── Show full detail on one result ────────────────────────────────────────────
print("\n--- Full detail ---")
detail = chain.invoke({"user_message": "Scale the auth-service deployment to 5 replicas."})
print(f"Intent    : {detail.intent}")
print(f"Confidence: {detail.confidence}")
print(f"Reasoning : {detail.reasoning}")
