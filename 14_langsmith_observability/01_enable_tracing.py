"""
Concept: Enabling LangSmith tracing — what it is and how to turn it on.

LangSmith is Anthropic/LangChain's observability platform for LLM apps.
It records every LLM call, tool invocation, chain step, and LangGraph node
as a "run tree" you can inspect in the web UI.

How it works:
  LangChain automatically traces when these env vars are set:
    LANGCHAIN_TRACING_V2=true          ← master switch
    LANGCHAIN_API_KEY=ls-...           ← your LangSmith API key
    LANGCHAIN_PROJECT=my-project       ← which project runs appear under
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  ← usually default

  No code changes required — tracing is injected via LangChain callbacks.

What gets recorded automatically:
  - Input and output of every LLM call
  - Prompt templates and rendered prompts
  - Token counts and latency
  - Tool call arguments and results
  - LangGraph node execution order and state diffs
  - Errors and tracebacks

Run:
    python 14_langsmith_observability/01_enable_tracing.py

Note: Set LANGCHAIN_TRACING_V2=true in your .env to see traces in the UI.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ── 1. Check tracing configuration ───────────────────────────────────────────

print("=== LangSmith configuration ===")
tracing  = os.getenv("LANGCHAIN_TRACING_V2", "false")
api_key  = os.getenv("LANGCHAIN_API_KEY", "")
project  = os.getenv("LANGCHAIN_PROJECT", "default")
endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

print(f"  LANGCHAIN_TRACING_V2 : {tracing}")
print(f"  LANGCHAIN_API_KEY    : {'set ✓' if api_key else 'NOT SET — traces will not be sent'}")
print(f"  LANGCHAIN_PROJECT    : {project}")
print(f"  LANGCHAIN_ENDPOINT   : {endpoint}")

if tracing.lower() != "true":
    print("\n  [INFO] Tracing is OFF. Calls below still work but won't appear in LangSmith.")
    print("         Set LANGCHAIN_TRACING_V2=true in .env to enable tracing.")
else:
    print("\n  [INFO] Tracing is ON. The call below will appear in LangSmith.")


# ── 2. A traced LLM call — no code change needed ─────────────────────────────

print("\n=== Simple traced LLM call ===")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

response = llm.invoke([
    SystemMessage(content="You are a concise assistant."),
    HumanMessage(content="What is LangSmith used for in one sentence?"),
])
print(f"Response: {response.content}")
print("\nIf tracing is ON, this call now appears in LangSmith under:")
print(f"  Project: {project}")
print("  URL: https://smith.langchain.com → Tracing → <your project>")


# ── 3. A traced LCEL chain ────────────────────────────────────────────────────

print("\n=== Traced LCEL chain ===")

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a DevOps assistant."),
        ("human",  "Explain '{concept}' in one sentence."),
    ])
    | llm
    | StrOutputParser()
)

result = chain.invoke({"concept": "blue-green deployment"})
print(f"Result: {result}")
print("\nIn LangSmith this shows as a nested run tree:")
print("  Chain (ChatPromptTemplate | ChatOpenAI | StrOutputParser)")
print("  └─ ChatPromptTemplate.format()")
print("  └─ ChatOpenAI.invoke()")
print("       tokens_used: input=XX  output=XX  total=XX")
print("  └─ StrOutputParser.parse()")


# ── 4. Adding metadata to a run ───────────────────────────────────────────────

print("\n=== Adding metadata and tags ===")
print("Pass config={'metadata': {...}, 'tags': [...]} to any invoke() call:")
print()

config = {
    "metadata": {
        "user_id":   "user-42",
        "session":   "sess-abc",
        "feature":   "devops-chat",
        "env":       "staging",
    },
    "tags": ["devops", "staging", "v1.2"],
}

result = chain.invoke({"concept": "canary release"}, config=config)
print(f"Result (with metadata): {result}")
print("\nIn LangSmith this run is filterable by:")
print("  - tag: 'devops'")
print("  - metadata.user_id: 'user-42'")
print("  - metadata.feature: 'devops-chat'")


# ── 5. Naming a run ───────────────────────────────────────────────────────────

print("\n=== Naming a run ===")
named_config = {
    "run_name": "explain-concept-v1",   # appears as the run name in LangSmith
    "metadata": {"concept": "canary"},
}

result = chain.invoke({"concept": "canary release"}, config=named_config)
print(f"Result (named run): {result}")
print("\nThe run appears in LangSmith with name 'explain-concept-v1'.")
print("Without run_name it defaults to the class name (e.g. 'RunnableSequence').")


# ── 6. What you see in the LangSmith UI ──────────────────────────────────────

print("\n=== What the LangSmith UI shows ===")
print("""
  For each run:
    Input        : the exact prompt sent to the model (rendered template)
    Output       : the model's response
    Latency      : time-to-first-token, total latency
    Token count  : input / output / total, and cost estimate
    Model        : gpt-4o-mini, etc.
    Feedback     : thumbs up/down ratings you add programmatically
    Metadata     : any key-value pairs you attached in config
    Tags         : string labels for filtering
    Run tree     : nested view of all sub-calls (chain → LLM → parser)

  For LangGraph graphs (section 14.02):
    Each node execution is a child run under the graph run.
    State input/output per node is recorded.
    Tool calls appear as nested children of the LLM node.
""")
