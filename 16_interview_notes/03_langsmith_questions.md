# LangSmith Interview Questions

---

## Q1. What is LangSmith and what are its three main use cases?

**Answer:**
LangSmith is LangChain's observability and evaluation platform for LLM applications. Its three main use cases are:

1. **Tracing / Debugging** — every LLM call, tool invocation, and graph node execution is recorded as a run tree. When a response is wrong, you open the trace and inspect exact inputs/outputs at each step without adding print statements.

2. **Evaluation** — run a dataset of `(input, expected_output)` pairs through your app, score each result with evaluator functions, and compare experiments side-by-side. Catch regressions before they reach production.

3. **Monitoring** — track token usage, cost, latency, and error rates in production. Annotators can rate responses (thumbs up/down). Aggregated feedback informs fine-tuning or prompt improvements.

---

## Q2. How do you enable LangSmith tracing? What code do you need to write?

**Answer:**
Zero code. Set these environment variables (in `.env`):

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=my-project
```

Every subsequent LangChain call — `llm.invoke()`, `chain.stream()`, `app.astream_events()` — is automatically traced via a callback handler that LangChain injects when it sees `LANGCHAIN_TRACING_V2=true`.

The only time you write code is to **add metadata/tags** (optional enrichment):
```python
result = app.invoke(input, config={
    "run_name":  "devops-agent",
    "metadata":  {"user_id": "u-42"},
    "tags":      ["production"],
})
```

---

## Q3. What does a LangSmith run tree look like for a ToolNode agent?

**Answer:**
```
LangGraph                             ← parent run (one per app.invoke())
├── llm_node
│     └── ChatOpenAI.invoke()
│           Input:  [SystemMessage, HumanMessage("Search Jira...")]
│           Output: AIMessage(tool_calls=[{name: "search_jira", args: {...}}])
│           Tokens: input=312  output=48  total=360
│           Cost:   $0.000054
│           Latency: 1.2s
├── tools (ToolNode)
│     └── search_jira
│           Input:  {"query": "login bugs"}
│           Output: "[JIRA] PROJ-101, PROJ-202"
│           Latency: 3ms
└── llm_node (second pass)
      └── ChatOpenAI.invoke()
            Input:  [System, Human, AIMessage(tool_calls), ToolMessage]
            Output: AIMessage("I found 2 Jira tickets...")
            Tokens: input=480  output=62  total=542
```

This lets you answer: did the tool get the right input? Did the LLM use the tool result correctly? Where did the latency come from?

---

## Q4. What is a LangSmith dataset and how do you add examples to it?

**Answer:**
A dataset is a named collection of `(input, expected_output)` pairs used for repeatable evaluation.

**Programmatically:**
```python
from langsmith import Client
client = Client()

dataset = client.create_dataset("devops-evals")
client.create_examples(
    inputs  = [{"question": "Create a Jira ticket"}, ...],
    outputs = [{"intent": "jira"}, ...],
    dataset_id = dataset.id,
)
```

**From the UI:** Open any production trace → click "Add to Dataset". This is the fastest way to capture real failure cases.

**From a JSONL file:** `client.create_examples_from_csv(...)` or by reading and iterating.

Best practice: seed the dataset with hand-crafted golden examples for each major intent, then grow it by adding real production failures.

---

## Q5. What is an LLM-as-judge evaluator and when should you use it?

**Answer:**
An LLM-as-judge evaluator uses a second LLM call to score the output of the first. It's the only way to evaluate open-ended qualities that can't be measured with exact match:

```python
from langsmith.evaluation import LangChainStringEvaluator

correctness_eval = LangChainStringEvaluator(
    "labeled_score_string",
    config={
        "criteria": {
            "correctness": "Is the response factually correct and complete?"
        },
        "normalize_by": 10,
    },
)
```

**Use it for:** tone, helpfulness, factual accuracy, safety, completeness.
**Don't use it for:** routing classification (use exact match), JSON structure (use schema validation).

Cost: ~$0.0001 per example with `gpt-4o-mini`. At 1000 examples, that's $0.10 per eval run — cheap enough for CI.

**Limitation:** LLM-as-judge has its own biases (prefers verbose answers, can miss subtle factual errors). Calibrate by comparing its scores against human annotations on a sample.

---

## Q6. How do you run evaluations in CI/CD?

**Answer:**
```python
# scripts/run_evals.py
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage

def target(inputs: dict) -> dict:
    result = app.invoke({"messages": [HumanMessage(content=inputs["question"])]})
    return {"answer": result["messages"][-1].content}

def intent_match(run, example) -> dict:
    return {
        "key":   "intent_correct",
        "score": 1 if run.outputs["intent"] == example.outputs["intent"] else 0,
    }

results = evaluate(
    target,
    data           = "devops-evals",
    evaluators     = [intent_match],
    experiment_prefix = f"ci-{git_sha[:8]}",
)

# Fail CI if accuracy drops
mean_score = results.to_pandas()["intent_correct"].mean()
if mean_score < 0.85:
    sys.exit(1)
```

Add to GitHub Actions with `LANGCHAIN_API_KEY` as a secret. Experiments are stored in LangSmith for side-by-side comparison across commits.

---

## Q7. What is `@traceable` and when is it necessary?

**Answer:**
`@traceable` wraps a plain Python function so it appears as a named span in the LangSmith trace tree:

```python
from langsmith import traceable

@traceable(name="preprocess-input", tags=["infra"])
def preprocess(raw: str) -> str:
    return raw.strip().lower()
```

**When it's necessary:**
- Pre/post-processing logic outside LangChain that you want to see in traces
- Business logic functions you want to attribute latency to
- Custom retrieval or caching layers
- Adding feedback to a specific sub-step via `get_current_run_tree().id`

Without `@traceable`, Python functions are invisible in LangSmith — you see the LangChain internals but not your own code.

---

## Q8. How would you use LangSmith to debug a regression where the agent started routing Jira questions to the AWS node?

**Answer:**
1. **Find the failing runs** — filter by `tag: production` and `metadata.feature: devops-assistant`, then filter by date range after the regression appeared.

2. **Open a trace** — look at the `classifier_node` run. Check its input (the user message) and output (the `RouteDecision` object). See what `intent` and `confidence` the classifier returned.

3. **Compare to a passing run** — open a pre-regression trace for a similar Jira question. Compare the classifier's structured output.

4. **Identify the cause** — likely candidates: a prompt change in the classifier, a model version change (`gpt-4o-mini` was updated), or the classifier schema was changed and the Jira description became less distinctive.

5. **Add to evaluation dataset** — "add to dataset" the failing examples.

6. **Fix and run eval** — update the classifier prompt and run `evaluate()` against the dataset. Confirm intent accuracy returns to baseline before merging the fix.

This workflow — trace → compare → dataset → eval — is why LangSmith exists.
