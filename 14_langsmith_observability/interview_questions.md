# Interview Questions — 14 LangSmith Observability

---

## Q1. What environment variables enable LangSmith tracing and what does each do?

**Answer:**

| Variable | Purpose | Required? |
|----------|---------|-----------|
| `LANGCHAIN_TRACING_V2` | Master switch — set to `"true"` to enable tracing | Yes |
| `LANGCHAIN_API_KEY` | LangSmith API key (starts with `ls-`) | Yes |
| `LANGCHAIN_PROJECT` | Which project runs are grouped under; defaults to `"default"` | No |
| `LANGCHAIN_ENDPOINT` | LangSmith API endpoint; defaults to `https://api.smith.langchain.com` | No |

No code change is needed — LangChain injects a callback handler automatically when these variables are set. Every `invoke()`, `stream()`, and `astream_events()` call is recorded.

---

## Q2. What does LangSmith record for each LLM call?

**Answer:**
For each LLM call LangSmith records:
- **Input**: the exact list of messages sent (rendered prompt template, not the raw template)
- **Output**: the full model response including `tool_calls` if present
- **Latency**: wall-clock time and time-to-first-token (for streaming)
- **Token counts**: input, output, and total tokens
- **Cost estimate**: dollars based on the model's public pricing
- **Model name**: `gpt-4o-mini`, etc.
- **Finish reason**: `"stop"`, `"tool_calls"`, `"length"`
- **Metadata and tags**: any key-value pairs and string labels you attached via `config`

---

## Q3. How do you attach metadata and tags to a run for later filtering in LangSmith?

**Answer:**
Pass a `config` dict with `"metadata"` and `"tags"` keys to any `invoke()` or `stream()` call:

```python
config = {
    "run_name":  "devops-agent-v1",      # display name in LangSmith
    "metadata":  {
        "user_id": "user-42",
        "feature": "devops-assistant",
        "env":     "production",
    },
    "tags": ["devops", "v1.2", "production"],
}

result = app.invoke(input_state, config=config)
```

In LangSmith you can then filter runs by `metadata.user_id = "user-42"` or tag `"production"`. This is essential for debugging a specific user's bad experience or aggregating costs per feature.

---

## Q4. What is `@traceable` and when do you use it?

**Answer:**
`@traceable` is a LangSmith decorator that wraps a plain Python function as a named span in the trace tree:

```python
from langsmith import traceable

@traceable(name="preprocess-message", tags=["preprocessing"])
def preprocess(raw: str) -> str:
    return raw.strip().lower()
```

Use it when:
- You have pre/post-processing logic outside LangChain that you want to appear in the trace.
- You want to track business logic (auth checks, rate limiting, caching) alongside LLM calls.
- You want to add feedback to a specific sub-step rather than the whole run.

Without `@traceable`, custom Python logic is invisible in LangSmith — you only see the LangChain/LangGraph internals.

---

## Q5. How does LangSmith represent a LangGraph run in its trace tree?

**Answer:**
Each `app.invoke()` creates a parent "LangGraph" run. Every graph node becomes a child run nested under it:

```
LangGraph (parent run)
├── __start__
├── llm_node
│     └── ChatOpenAI.invoke()   ← prompt, response, tokens, latency
├── tools
│     ├── search_jira           ← args: {"query": "..."}, result: "..."
│     └── get_aws_cost          ← args: {"month": "2026-06"}, result: "..."
└── llm_node (second pass)
      └── ChatOpenAI.invoke()   ← final answer
```

This lets you debug exactly which node received which state and what it produced — without adding any print statements.

---

## Q6. What is a LangSmith dataset and how do you build one?

**Answer:**
A dataset is a collection of `(input, expected_output)` example pairs used for systematic evaluation.

Three ways to build one:
1. **Hand-crafted** — write examples that cover important cases and known failure modes.
2. **From production traces** — in the LangSmith UI, click "Add to Dataset" on any run. This seeds the dataset with real user inputs.
3. **Programmatically** via `client.create_examples()` with a list of input/output dicts.

The dataset is reusable: run multiple experiments against the same dataset to compare prompt versions, models, or routing logic.

---

## Q7. What are the four main evaluator types in LangSmith and when do you use each?

**Answer:**

| Evaluator | When to use |
|-----------|-------------|
| **Exact match** | Classification output, routing decisions, structured fields with one correct answer |
| **Regex / keyword** | Format checks (date formats, ID patterns), presence of required terms |
| **Embedding distance** | Semantic equivalence — the answer says the same thing in different words |
| **LLM-as-judge** | Open-ended quality: tone, helpfulness, factual accuracy, safety |

Prefer exact match when possible (deterministic and free). Reserve LLM-as-judge for outputs that require semantic understanding. At 1000 examples with `gpt-4o-mini`, an LLM-as-judge eval costs ~$1 per run.

---

## Q8. How do you integrate LangSmith evaluations into a CI/CD pipeline?

**Answer:**
1. Add `LANGCHAIN_API_KEY` and `OPENAI_API_KEY` as CI secrets.
2. Write a script that calls `evaluate(target=run_agent, data="dataset-name", evaluators=[...])`.
3. Parse the aggregate scores from the results object.
4. Fail the build if a key metric drops below a threshold (e.g., intent accuracy < 85%).

```yaml
# GitHub Actions step
- name: Run evals and assert quality
  env:
    LANGCHAIN_TRACING_V2: true
    LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
  run: |
    python scripts/run_evals.py
    python scripts/assert_thresholds.py  # exits 1 if score < threshold
```

This prevents merging changes that regress quality. The experiment results are stored in LangSmith for side-by-side comparison with previous runs.
