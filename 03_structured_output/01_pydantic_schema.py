"""
Concept: Defining output schemas with Pydantic.

LangChain uses Pydantic models as the blueprint for structured output.
The model reads the class name, field names, types, and docstrings to
understand exactly what JSON it should produce.

Good schema design rules:
  - Field names should be self-explanatory (the model reads them).
  - Add a Field(description=...) for any field that needs clarification.
  - Use Optional[T] for fields that may not always be present.
  - Keep schemas flat when possible — nested schemas work but cost more tokens.

Run:
    python 03_structured_output/01_pydantic_schema.py
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


# ── Simple schema ─────────────────────────────────────────────────────────────

class MovieReview(BaseModel):
    """A structured review of a movie."""

    title: str = Field(description="The title of the movie.")
    rating: int = Field(description="Rating out of 10.", ge=1, le=10)
    sentiment: Literal["positive", "negative", "mixed"]
    summary: str = Field(description="One-sentence summary of the review.")


# ── Schema with optional fields ───────────────────────────────────────────────

class SupportTicket(BaseModel):
    """A parsed customer support request."""

    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high", "critical"]
    title: str = Field(description="Short title for the ticket (max 10 words).")
    description: str = Field(description="Full description of the issue.")
    affected_service: Optional[str] = Field(
        default=None,
        description="The product or service affected, if mentioned.",
    )


# ── Inspect the JSON schema LangChain will send to the model ─────────────────

import json

print("=== MovieReview JSON schema ===")
print(json.dumps(MovieReview.model_json_schema(), indent=2))

print("\n=== SupportTicket JSON schema ===")
print(json.dumps(SupportTicket.model_json_schema(), indent=2))

# ── Validate an instance manually ─────────────────────────────────────────────
review = MovieReview(
    title="Inception",
    rating=9,
    sentiment="positive",
    summary="A mind-bending thriller that rewards attention.",
)
print("\n=== Valid instance ===")
print(review.model_dump())

# Pydantic catches bad values at construction time
try:
    MovieReview(title="X", rating=11, sentiment="positive", summary="too high")
except Exception as e:
    print("\n=== Validation error (rating=11) ===")
    print(e)
