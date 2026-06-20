"""
Concept: LangSmith Evaluation — testing LLM applications systematically.

Unit tests check that code paths run without errors.
LangSmith evals check that LLM responses are actually GOOD:
  - Is the answer factually correct?
  - Is the right tool being called?
  - Is the tone appropriate?
  - Does the output match the expected schema?

Key concepts:
  Dataset   : a collection of (input, expected_output) example pairs.
              Built from: hand-crafted examples, production traces, golden sets.
  Evaluator : a function that scores a run's output.
              Types: exact match, LLM-as-judge, semantic similarity, custom.
  Experiment: one run of evaluate() — runs all dataset examples through your
              app, calls evaluators on each result, aggregates scores.

This file is a concept sketch — it prints patterns with explanation.
Running actual evals requires LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2=true.

Run:
    python 14_langsmith_observability/03_evaluation_overview.py
"""


def section(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


# ═══════════════════════════════════════════════════════════════════════════════
section("1. Creating a dataset")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  from langsmith import Client

  client = Client()

  # Create a named dataset
  dataset = client.create_dataset(
      dataset_name="devops-assistant-evals",
      description="Golden test set for the DevOps assistant",
  )

  # Add examples: (input, expected_output) pairs
  examples = [
      {
          "inputs":  {"question": "Create a Jira ticket for the login bug"},
          "outputs": {"intent": "jira", "confidence_min": 0.9},
      },
      {
          "inputs":  {"question": "What is my AWS cost for June 2026?"},
          "outputs": {"intent": "aws"},
      },
      {
          "inputs":  {"question": "Scale the web deployment to 5 replicas"},
          "outputs": {"intent": "kubernetes"},
      },
      {
          "inputs":  {"question": "What is a blue-green deployment?"},
          "outputs": {"intent": "general"},
      },
  ]

  client.create_examples(
      inputs  = [e["inputs"]  for e in examples],
      outputs = [e["outputs"] for e in examples],
      dataset_id = dataset.id,
  )

  # Or add examples from a production trace (click "Add to Dataset" in the UI)
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("2. Writing evaluators")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  Evaluators are plain Python functions with this signature:
    def my_eval(run, example) -> dict:
        ...
        return {"key": "metric-name", "score": 0.0-1.0}

  ── a) Exact match evaluator ──

  def intent_match(run, example) -> dict:
      predicted = run.outputs.get("intent")
      expected  = example.outputs.get("intent")
      return {
          "key":   "intent_correct",
          "score": 1 if predicted == expected else 0,
      }


  ── b) Keyword in output evaluator ──

  def answer_mentions_jira(run, example) -> dict:
      answer = run.outputs.get("answer", "")
      return {
          "key":   "mentions_jira",
          "score": 1 if "PROJ-" in answer else 0,
      }


  ── c) LLM-as-judge evaluator (built-in) ──

  from langsmith.evaluation import LangChainStringEvaluator

  # Uses an LLM to judge if the answer is correct
  correctness_eval = LangChainStringEvaluator(
      "labeled_score_string",
      config={
          "criteria": {
              "correctness": "Is the answer factually correct given the context?"
          },
          "normalize_by": 10,
      },
  )

  # Built-in evaluators available:
  # - "labeled_score_string": LLM scores 1-10, normalised
  # - "labeled_criteria":     LLM returns Y/N per criterion
  # - "embedding_distance":   cosine similarity to reference answer
  # - "exact_match":          string equality
  # - "regex_match":          answer matches a pattern
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("3. Running an experiment with evaluate()")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  from langsmith.evaluation import evaluate

  # The target function wraps your app
  def run_agent(inputs: dict) -> dict:
      result = app.invoke(
          {"messages": [HumanMessage(content=inputs["question"])]},
          config={"configurable": {"thread_id": "eval-thread"}},
      )
      return {
          "answer": result["messages"][-1].content,
          "intent": result.get("intent", "unknown"),
      }

  # Run the experiment
  results = evaluate(
      target          = run_agent,
      data            = "devops-assistant-evals",   # dataset name or id
      evaluators      = [intent_match, answer_mentions_jira, correctness_eval],
      experiment_prefix = "devops-v1",              # appears in LangSmith UI
      num_repetitions   = 1,                        # run each example N times
      max_concurrency   = 4,                        # parallel eval workers
  )

  # Results object has aggregate scores
  print(results.to_pandas())   # requires pandas

  # Output example:
  #   example_id | intent_correct | mentions_jira | correctness
  #   abc-001    |            1.0 |           1.0 |         0.9
  #   abc-002    |            0.0 |           0.0 |         0.4
  #   ...
  #   Mean       |           0.75 |          0.80 |        0.78
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("4. Viewing results in the LangSmith UI")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  After evaluate() runs:
    smith.langchain.com → Datasets & Testing → devops-assistant-evals
      → Experiments tab
        → devops-v1           ← your experiment
          Aggregate scores:
            intent_correct : 0.75
            mentions_jira  : 0.80
            correctness    : 0.78

          Per-example breakdown:
            Example 1: ✓ intent_correct | ✓ mentions_jira | 0.9 correctness
            Example 2: ✗ intent_correct | ✗ mentions_jira | 0.4 correctness
              └─ Click to see the full run: input, output, evaluator reasoning

  You can compare multiple experiments side-by-side to track improvement:
    devops-v1  →  devops-v2 (after prompt fix)  →  devops-v3 (after model upgrade)
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("5. Online evaluation — scoring production runs")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  Offline eval: run a fixed dataset through your app in a test environment.
  Online eval:  score production runs as they happen, using the same evaluators.

  ── Programmatic feedback on a live run ──

  from langsmith import Client
  from langsmith.run_helpers import get_current_run_tree

  client = Client()

  @traceable
  def run_agent_with_feedback(question: str, expected_intent: str) -> str:
      result = app.invoke({"messages": [HumanMessage(content=question)]})
      answer = result["messages"][-1].content

      run = get_current_run_tree()
      score = 1 if expected_intent in answer.lower() else 0
      client.create_feedback(
          run_id  = run.id,
          key     = "intent_in_answer",
          score   = score,
      )
      return answer

  ── Human-in-the-loop annotation ──
  In the LangSmith UI, annotators can open any run and:
  - Click thumbs up / thumbs down
  - Add a free-text comment
  - Tag the run (e.g., "hallucination", "off-topic")
  These annotations aggregate as feedback and can trigger fine-tuning.
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("6. Evaluation workflow in a CI/CD pipeline")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  # .github/workflows/eval.yml (concept)
  # Runs on every PR that touches the prompt or agent code

  steps:
    - name: Run LangSmith evals
      env:
        LANGCHAIN_API_KEY:      ${{ secrets.LANGCHAIN_API_KEY }}
        LANGCHAIN_TRACING_V2:   true
        LANGCHAIN_PROJECT:      ci-evals
        OPENAI_API_KEY:         ${{ secrets.OPENAI_API_KEY }}
      run: python scripts/run_evals.py

    - name: Assert quality threshold
      run: |
        python - <<'EOF'
        import json, sys
        results = json.load(open("eval_results.json"))
        intent_acc = results["intent_correct"]["mean"]
        if intent_acc < 0.85:
            print(f"FAIL: intent accuracy {intent_acc:.0%} < 85% threshold")
            sys.exit(1)
        print(f"PASS: intent accuracy {intent_acc:.0%}")
        EOF

  This gate prevents merging a PR that regresses intent routing below 85%.
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("7. Summary: when to use each eval type")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  ┌──────────────────────┬────────────────────────────────────────────────┐
  │ Evaluator type       │ Best for                                       │
  ├──────────────────────┼────────────────────────────────────────────────┤
  │ Exact match          │ Classification, routing, structured output      │
  │ Regex match          │ Format checks (dates, IDs, JSON structure)     │
  │ Embedding distance   │ Paraphrase / semantic equivalence              │
  │ LLM-as-judge         │ Tone, helpfulness, factual accuracy            │
  │ Human annotation     │ Nuanced quality, edge cases, fine-tuning data  │
  └──────────────────────┴────────────────────────────────────────────────┘

  Cost note:
    LLM-as-judge costs ~$0.001 per example with gpt-4o-mini.
    At 1000 examples, that is ~$1 per eval run — cheap enough to run in CI.
    Use exact match where possible; reserve LLM-judge for open-ended outputs.
""")
