# 14 — LangSmith Observability

> Status: **Complete**

## What This Section Covers

- Enabling LangSmith tracing via environment variables
- Debugging graph execution in the LangSmith UI
- `@traceable` for custom Python functions
- Evaluation: datasets, evaluators, experiments, CI/CD integration

## Files

| File | Purpose |
|------|---------|
| `01_enable_tracing.py` | Env var check, first traced LLM call, traced LCEL chain, metadata/tags, run naming |
| `02_trace_a_graph.py` | Full LangGraph trace with run_name/metadata/tags, multi-turn tracing, `@traceable`, feedback API, per-node breakdown |
| `03_evaluation_overview.py` | Datasets, evaluator types (exact/regex/embedding/LLM-judge), `evaluate()`, CI/CD eval gate — concept only, no API key needed |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
# Enable tracing first (in .env):
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=ls-...
# LANGCHAIN_PROJECT=langchain-learning

python 14_langsmith_observability/01_enable_tracing.py
python 14_langsmith_observability/02_trace_a_graph.py
python 14_langsmith_observability/03_evaluation_overview.py  # no API key needed
```

## Key Concepts

**Zero-code tracing** — set `LANGCHAIN_TRACING_V2=true` and every LangChain/LangGraph call is automatically recorded. No code changes needed.

**Run metadata** — pass `config={"run_name": ..., "metadata": {...}, "tags": [...]}` to any `invoke()` for searchable, labelled traces.

**`@traceable`** — wraps plain Python functions as trace spans so pre/post-processing logic is visible alongside LLM calls.

**Datasets + evaluate()** — a `(input, expected_output)` dataset plus evaluator functions gives you repeatable, quantitative quality metrics.

**CI/CD gate** — run `evaluate()` in CI and fail the build if intent accuracy drops below threshold. Prevents prompt regressions from reaching production.

## LangSmith Trace Structure for a LangGraph Run

```
LangGraph (parent run)         ← one per app.invoke()
├── llm_node
│     └── ChatOpenAI.invoke()  ← prompt, response, tokens, cost, latency
├── tools
│     ├── search_jira          ← tool args + result
│     └── get_aws_cost
└── llm_node (second pass)
      └── ChatOpenAI.invoke()  ← final answer
```
