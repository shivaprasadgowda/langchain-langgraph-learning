# Interview Questions — 03 Structured Output

---

## Q1. What is `with_structured_output` and how does it work internally?

**Answer:**
`llm.with_structured_output(Schema)` returns a new `Runnable` that:

1. Converts the Pydantic model to a JSON schema and passes it to the model as a **tool definition**.
2. Instructs the model to respond by "calling" that tool — i.e. producing JSON that matches the schema.
3. Validates the JSON response against the Pydantic model.
4. Returns a typed Python instance, not a string.

This is more reliable than asking the model to produce JSON in plain text because the model is guided by the tool-call mechanism rather than freeform generation.

---

## Q2. Why use `Literal[...]` for categorical fields instead of `str`?

**Answer:**
`Literal["bug", "task", "story"]` constrains the model to exactly those values. If you use `str`, the model may invent its own labels ("Bug Report", "feature", "BUG") that break downstream `if/elif` logic.

`Literal` also:
- Documents the allowed values directly in the schema the model reads.
- Causes Pydantic to raise a `ValidationError` if the model somehow returns an unexpected value.
- Maps cleanly to routing logic — `if ticket.ticket_type == "bug"`.

---

## Q3. What is the difference between structured output and an output parser like `JsonOutputParser`?

**Answer:**

| Approach | Mechanism | Reliability |
|----------|-----------|-------------|
| `with_structured_output` | Uses the model's tool-call API. Model is **forced** to produce valid JSON. | High |
| `JsonOutputParser` | Asks the model to produce JSON in plain text, then parses it. | Lower — model can produce invalid JSON or wrap it in markdown |

`with_structured_output` is the modern preferred approach for any provider that supports tool calling (OpenAI, Anthropic, Google). Output parsers are a fallback for models that don't.

---

## Q4. When should you set `temperature=0` for structured output tasks?

**Answer:**
Always. Classification, extraction, and routing have one correct answer (or a small set of equally valid answers). Randomness (`temperature > 0`) only adds noise — the model might choose a different label on the next identical call. `temperature=0` makes the output deterministic, which is critical for testing and production reliability.

---

## Q5. How do you handle fields that may not be present in the source text?

**Answer:**
Mark them `Optional[T]` with a `default=None`:

```python
assignee: Optional[str] = Field(
    default=None,
    description="Person mentioned as responsible, if any.",
)
```

The model sets these to `None` when the information is absent. Without `Optional`, the model is forced to invent a value, which produces hallucinations.

For lists that may be empty use `list[str] = Field(default_factory=list)`.

---

## Q6. What role does the `Field(description=...)` play?

**Answer:**
The description is included in the JSON schema sent to the model. The model reads it when deciding what value to put in the field. Without descriptions, the model guesses based on field names alone — which works for obvious names but fails for ambiguous ones like `priority` (1–5? low/medium/high? P0–P4?).

A good description:
- States the allowed values if they aren't already expressed via `Literal`.
- Explains the inference rule ("inferred from urgency cues").
- Gives the expected format or length ("max 10 words").

---

## Q7. How does the router pattern in file 05 connect to LangGraph?

**Answer:**
In LangGraph, `add_conditional_edges(node, router_fn)` calls `router_fn` on the graph state after a node runs and uses the returned string to decide which node to go to next.

The pattern is:
```python
# Router function reads the structured output from state
def router(state) -> str:
    return state["route_decision"].destination  # e.g. "jira_node"

graph.add_conditional_edges("classifier_node", router)
```

Because `destination` is `Literal["jira_node", "aws_node", ...]`, the returned string is guaranteed to be a valid node name. No string parsing or `.lower()` gymnastics needed. Section 07 implements this fully.

---

## Q8. Can `with_structured_output` handle nested Pydantic models?

**Answer:**
Yes. You can nest models:

```python
class Address(BaseModel):
    city: str
    country: str

class Person(BaseModel):
    name: str
    address: Address
```

The JSON schema is recursively generated and the model fills in the nested structure. However, deeply nested schemas increase prompt complexity and token cost. Prefer flat schemas when possible and only nest when the hierarchy genuinely reflects the domain.
