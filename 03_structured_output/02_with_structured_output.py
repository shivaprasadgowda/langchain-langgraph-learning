"""
Concept: with_structured_output — bind a Pydantic schema to the LLM.

llm.with_structured_output(Schema) returns a new Runnable that:
  1. Sends the JSON schema to the model as a tool definition.
  2. Forces the model to respond by calling that tool.
  3. Validates the response against the Pydantic model.
  4. Returns a typed Python object, not a string.

This is the modern LangChain approach (preferred over output parsers
that ask the model to produce JSON in plain text, which is fragile).

Run:
    python 03_structured_output/02_with_structured_output.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# ── Schema ────────────────────────────────────────────────────────────────────

class MovieReview(BaseModel):
    """A structured review of a movie."""
    title: str = Field(description="The movie title.")
    rating: int = Field(description="Rating from 1 to 10.", ge=1, le=10)
    sentiment: Literal["positive", "negative", "mixed"]
    summary: str = Field(description="One-sentence summary.")


# ── Bind schema to the LLM ────────────────────────────────────────────────────
# temperature=0 — extraction tasks have one correct answer
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(MovieReview)

# ── Simple invocation ─────────────────────────────────────────────────────────
review_text = (
    "Inception is a masterpiece. Nolan crafts a visually stunning and "
    "intellectually engaging thriller. Easily a 9 out of 10."
)

result = structured_llm.invoke(f"Extract a structured review from: {review_text}")

# result is a MovieReview instance — fully typed and validated
print(type(result))          # <class 'MovieReview'>
print(result.title)          # Inception
print(result.rating)         # 9
print(result.sentiment)      # positive
print(result.model_dump())   # dict representation

# ── Inside a LCEL chain ───────────────────────────────────────────────────────
template = ChatPromptTemplate.from_messages([
    ("system", "You extract structured movie reviews from text."),
    ("human", "Extract a review from this text:\n\n{text}"),
])

chain = template | structured_llm

result2 = chain.invoke({
    "text": (
        "The Dark Knight is an intense crime drama with Heath Ledger "
        "delivering an iconic performance. It's a solid 10 for me, "
        "though some scenes feel too long — mixed feelings overall."
    )
})

print("\n--- Second review ---")
print(result2.model_dump())
