"""
Concept: Entity / field extraction with structured output.

Extraction pulls specific facts out of free-form text and maps them
to a schema. The model fills in what it can find and leaves Optional
fields as None when information is missing.

Real-world uses: parsing emails, support messages, form inputs,
meeting transcripts, or any natural-language source.

Run:
    python 03_structured_output/04_ticket_extraction.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, Literal

load_dotenv()


# ── Schema ────────────────────────────────────────────────────────────────────

class JiraTicket(BaseModel):
    """A Jira ticket extracted from a user's free-form request."""

    title: str = Field(description="Short ticket title, max 10 words.")
    description: str = Field(description="Full description of the issue or task.")
    ticket_type: Literal["bug", "task", "story", "improvement"] = Field(
        description="The type of Jira issue."
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Priority level inferred from urgency cues in the message."
    )
    assignee: Optional[str] = Field(
        default=None,
        description="Person mentioned as responsible, if any.",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Relevant labels inferred from the message (e.g. 'auth', 'frontend').",
    )


# ── Chain ─────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You extract structured Jira ticket fields from user messages. "
        "Infer missing fields from context. Leave optional fields None if "
        "there is genuinely no information.",
    ),
    ("human", "Extract a Jira ticket from this message:\n\n{message}"),
])

chain = template | llm.with_structured_output(JiraTicket)


# ── Examples ──────────────────────────────────────────────────────────────────

messages = [
    # Rich message — most fields are present
    (
        "Hey, the login page is totally broken for SSO users since the deploy "
        "last night. Users are getting a 401. This is critical, please assign "
        "it to Sarah on the auth team."
    ),
    # Sparse message — many fields must be inferred
    (
        "We should add dark mode to the dashboard at some point."
    ),
    # Task with urgency
    (
        "Need to upgrade the postgres version on the prod DB before the "
        "compliance audit on Friday. Medium priority I think."
    ),
]

for msg in messages:
    ticket: JiraTicket = chain.invoke({"message": msg})
    print("Input    :", msg[:70] + ("..." if len(msg) > 70 else ""))
    print("Title    :", ticket.title)
    print("Type     :", ticket.ticket_type)
    print("Priority :", ticket.priority)
    print("Assignee :", ticket.assignee)
    print("Labels   :", ticket.labels)
    print()
