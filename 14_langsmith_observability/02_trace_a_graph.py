"""
Concept: Tracing a LangGraph graph with LangSmith.

When LANGCHAIN_TRACING_V2=true, every app.invoke() / app.stream() call
is automatically recorded as a parent run in LangSmith. Each graph node
becomes a child run nested under it.

The trace structure for a ToolNode agent looks like:
  LangGraph (graph run)
  ├─ __start__
  ├─ llm_node
  │    └─ ChatOpenAI.invoke()  ← LLM call with prompt + response
  ├─ tools
  │    ├─ search_jira          ← tool execution with args + result
  │    └─ get_aws_cost
  └─ llm_node (second pass)
       └─ ChatOpenAI.invoke()  ← final answer

Advanced tracing features used here:
  - run_name    : human-readable label in LangSmith
  - metadata    : searchable key-value pairs on the run
  - tags        : string labels for filtering
  - @traceable  : wrap any plain Python function as a LangSmith span

Run:
    python 14_langsmith_observability/02_trace_a_graph.py
"""

import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langsmith import traceable

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
PROJECT  = os.getenv("LANGCHAIN_PROJECT", "default")


# ── Graph definition ──────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def search_jira(query: str) -> str:
    """Search Jira tickets by keyword."""
    return f"[JIRA] '{query}': PROJ-101 (Login bug, open), PROJ-202 (API timeout, in-progress)"


@tool
def get_aws_cost(month: str) -> str:
    """Get AWS cost summary for a month (YYYY-MM)."""
    return f"[AWS] {month}: EC2 $312, RDS $98, S3 $15 → Total $425"


tools = [search_jira, get_aws_cost]
llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def llm_node(state: State) -> dict:
    system = SystemMessage(content="You are a DevOps assistant with Jira and AWS tools.")
    return {"messages": [llm.invoke([system] + state["messages"])]}

builder = StateGraph(State)
builder.add_node("llm",   llm_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)


# ── 1. Basic traced invocation ────────────────────────────────────────────────

print("=== 1. Basic traced graph invocation ===")
print(f"Tracing: {'ON → run will appear in LangSmith' if TRACING else 'OFF → local only'}")
print(f"Project: {PROJECT}\n")

config = {
    "configurable": {"thread_id": "langsmith-demo-001"},
    "run_name": "devops-agent-v1",       # name shown in LangSmith
    "metadata": {
        "user_id":   "demo-user",
        "session":   "langsmith-demo",
        "feature":   "devops-assistant",
    },
    "tags": ["demo", "devops", "gpt-4o-mini"],
}

result = app.invoke(
    {"messages": [HumanMessage(content="Search Jira for login issues.")]},
    config=config,
)

print(f"Answer: {result['messages'][-1].content}")
print(f"\nIf tracing is ON, visit https://smith.langchain.com")
print(f"→ Project '{PROJECT}' → Run 'devops-agent-v1'")
print(f"→ You'll see: graph run → llm_node → tools node → llm_node (final)")


# ── 2. Thread-scoped tracing (multi-turn) ─────────────────────────────────────

print("\n=== 2. Multi-turn conversation (same thread_id) ===")

config2 = {
    "configurable": {"thread_id": "langsmith-demo-001"},  # same thread → continues
    "run_name": "devops-agent-v1-turn2",
    "metadata": {"user_id": "demo-user", "turn": 2},
}

result2 = app.invoke(
    {"messages": [HumanMessage(content="Now get the AWS cost for June 2026.")]},
    config=config2,
)
print(f"Answer: {result2['messages'][-1].content}")
print("\nIn LangSmith both turns appear as separate runs linked by thread_id.")


# ── 3. @traceable — wrap plain Python as a LangSmith span ────────────────────

print("\n=== 3. @traceable on plain Python functions ===")

@traceable(
    name="preprocess-user-message",     # name shown in the trace tree
    tags=["preprocessing"],
    metadata={"version": "1"},
)
def preprocess(raw: str) -> str:
    """Trim, lowercase, and add context prefix."""
    cleaned = raw.strip().lower()
    return f"[user query] {cleaned}"


@traceable(name="postprocess-response", tags=["postprocessing"])
def postprocess(response: str, user_id: str) -> dict:
    """Format final response for the API."""
    return {
        "user_id":   user_id,
        "answer":    response,
        "chars":     len(response),
        "truncated": len(response) > 500,
    }


raw_input = "  Search Jira for deployment failures  "
processed = preprocess(raw_input)
print(f"Preprocessed: {processed!r}")

config3 = {
    "configurable": {"thread_id": "langsmith-demo-002"},
    "run_name": "traced-pipeline",
}
r = app.invoke({"messages": [HumanMessage(content=processed)]}, config=config3)
final = r["messages"][-1].content
output = postprocess(final, user_id="demo-user")
print(f"Postprocessed output: {output}")
print("\nAll three functions (preprocess, graph, postprocess) appear")
print("as sibling spans in the same LangSmith trace when called inside")
print("a traced parent context.")


# ── 4. Adding human feedback programmatically ─────────────────────────────────

print("\n=== 4. Adding feedback to a run (concept) ===")
print("""
  from langsmith import Client

  client = Client()

  # After a user rates the response (e.g., thumbs up in the UI):
  client.create_feedback(
      run_id   = run_id,          # the run_id from the LangSmith response
      key      = "user_rating",   # feedback dimension name
      score    = 1,               # 1 = positive, 0 = negative
      comment  = "Correct and concise",
  )

  # For automated correctness scoring:
  client.create_feedback(
      run_id = run_id,
      key    = "correctness",
      score  = 0.9,   # from your evaluator (0.0–1.0)
  )

  Feedback appears in LangSmith under the run's "Feedback" tab and can
  be aggregated across thousands of runs to track quality over time.
""")


# ── 5. What LangSmith shows per node ──────────────────────────────────────────

print("=== 5. What LangSmith records per graph node ===")
print("""
  llm_node run:
    Input  : [SystemMessage, HumanMessage(s)]
    Output : AIMessage with .content or .tool_calls
    Latency: time-to-first-token, total
    Tokens : input_tokens=X, output_tokens=Y, total_tokens=Z
    Cost   : $0.000N (based on model pricing)

  tools node (ToolNode):
    Input  : AIMessage with tool_calls=[{name, args}]
    Per-tool child run:
      search_jira:
        Input : {"query": "login issues"}
        Output: "[JIRA] ..."
        Latency: N ms (pure Python → <1ms)

  Debugging workflow:
    1. Find a bad response in production
    2. Open the LangSmith run for that request
    3. Inspect each node's input/output to find where reasoning went wrong
    4. Check tool outputs — was the Jira search returning wrong tickets?
    5. Add test case to evaluation dataset (section 03)
    6. Fix prompt or routing logic and re-run the eval
""")
