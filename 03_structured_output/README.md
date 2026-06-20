# 03 — Structured Output

> Status: **Complete**

## What This Section Covers

- Defining output schemas with Pydantic
- `with_structured_output`
- Classification use-case
- Ticket extraction use-case
- Router-ready structured output

## Files

| File | Purpose |
|------|---------|
| `01_pydantic_schema.py` | Define Pydantic models, inspect JSON schema, test validation |
| `02_with_structured_output.py` | Bind a schema to the LLM; use in a standalone call and in a LCEL chain |
| `03_classification.py` | Intent classification with `Literal` labels and confidence scores |
| `04_ticket_extraction.py` | Extract Jira ticket fields from free-form messages, including `Optional` fields |
| `05_router_output.py` | Full routing pipeline: classifier → `RouteDecision` → router fn → specialist node |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
python 03_structured_output/01_pydantic_schema.py
python 03_structured_output/02_with_structured_output.py
python 03_structured_output/03_classification.py
python 03_structured_output/04_ticket_extraction.py
python 03_structured_output/05_router_output.py
```

## Key Concepts

**`with_structured_output(Schema)`** — uses the model's tool-call API to force valid JSON; returns a typed Pydantic instance, not a string.

**`Literal[...]` fields** — constrain the model to exact label values; prevents hallucinated categories and maps cleanly to routing logic.

**`Optional[T] = Field(default=None)`** — lets the model leave a field blank when the source text doesn't contain that information.

**`temperature=0`** — always use for classification and extraction; these tasks have one correct answer.

**Router pattern** — `destination: Literal["node_a", "node_b"]` maps directly to LangGraph node names; the router function is just `return state.destination`. Wired into a real graph in section 07.
