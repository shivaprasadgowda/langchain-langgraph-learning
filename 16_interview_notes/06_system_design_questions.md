# Production System Design Questions

---

## Q1. Design a multi-tenant AI chat system for a DevOps platform.

**Requirements:** multiple companies, each with their own data and rate limits; multi-turn conversations; streaming responses; tool calling for Jira, AWS, and Kubernetes.

**Answer:**

```
┌──────────────────────────────────────────────────────────────────┐
│  React frontend (per company, white-label)                       │
│  JWT carries: user_id, company_id, scopes                       │
└─────────────────────┬────────────────────────────────────────────┘
                      │ HTTPS / SSE
┌─────────────────────▼────────────────────────────────────────────┐
│  FastAPI (stateless, N instances)                                │
│  Middleware:                                                      │
│    1. JWT verify → extract user_id, company_id                  │
│    2. Redis rate limit: key="rate:{company_id}:{user_id}"       │
│    3. Input guardrail (length, injection)                        │
│    4. Thread scoping: thread_id = "{company_id}:{user_id}:{uuid}"│
└──────────┬──────────────────┬──────────────────┬────────────────┘
           │                  │                  │
   LangGraph agent      PostgreSQL          Redis
   (per company tools)  (checkpoints,       (rate limits,
                         namespace by        session cache)
                         company_id)
```

**Multi-tenancy isolation:**
- `thread_id` prefixed with `company_id` — no accidental cross-tenant state
- Each company's tools configured separately (different Jira base URLs, AWS accounts, K8s clusters)
- Rate limits keyed on `company_id:user_id` — per-company quotas
- PostgreSQL row-level security: `WHERE company_id = $1` on all checkpoint reads/writes
- LangSmith: separate project per company environment

**Tool isolation:**
```python
def build_agent_for_company(company: Company) -> CompiledGraph:
    tools = [
        JiraTool(base_url=company.jira_url, api_key=company.jira_key),
        AWSTool(account_id=company.aws_account, role=company.aws_role),
    ]
    return build_and_compile_graph(tools, checkpointer)
```

---

## Q2. How would you build a self-corrective RAG system that retries retrieval if the context is insufficient?

**Answer:**

```
START → retrieve → grade_docs → generate → grade_output → END
                       ↓ (bad docs)            ↓ (bad answer)
                   rewrite_query           retrieve (again)
                       ↓
                   retrieve
```

**Grade docs node:** classifies whether retrieved chunks are relevant:
```python
class DocGrade(BaseModel):
    relevant: bool
    reason:   str

grader = llm.with_structured_output(DocGrade)

def grade_docs_node(state: State) -> dict:
    question = state["messages"][-1].content
    docs     = state["context_docs"]
    results  = [grader.invoke([
        SystemMessage("Are these docs relevant to the question?"),
        HumanMessage(f"Question: {question}\n\nDoc: {doc.page_content}"),
    ]) for doc in docs]
    relevant = [d for d, r in zip(docs, results) if r.relevant]
    return {"context_docs": relevant, "retrieval_attempts": state.get("retrieval_attempts", 0) + 1}

def route_after_grading(state: State) -> str:
    if len(state["context_docs"]) >= 2:
        return "generate"
    if state.get("retrieval_attempts", 0) >= 2:
        return "generate"   # give up and try with what we have
    return "rewrite_query"
```

**Query rewriter node:**
```python
def rewrite_query_node(state: State) -> dict:
    original = state["messages"][-1].content
    rewritten = llm.invoke([
        SystemMessage("Rewrite this query to be more specific for document retrieval."),
        HumanMessage(original),
    ])
    return {"rewritten_query": rewritten.content}
```

**Max retries guard:** `retrieval_attempts >= 2` breaks the loop. Without it, a query about missing information would loop forever.

---

## Q3. Design a production HITL system for approving AI-generated Jira tickets.

**Requirements:** agent creates Jira tickets; a human must approve each; approvals can happen asynchronously (user may approve hours later); the system must survive server restarts between the request and the approval.

**Answer:**

**Why this is hard without LangGraph:** a `while True` loop can't survive a server restart. The pause state lives in memory.

**Solution: LangGraph + PostgreSQL checkpointer:**

```
POST /create-ticket
  → app.invoke(input, config) with interrupt_before=["tools"]
  → graph pauses after LLM generates tool_calls
  → checkpoint written to PostgreSQL
  → API returns {"status": "pending_approval", "thread_id": "...", "pending_tools": [...]}
  → 200 OK returned immediately

... hours later ...

User opens approval UI, clicks Approve/Reject.

POST /approve {"thread_id": "...", "approved": true}
  → FastAPI reads checkpoint from PostgreSQL (any instance can handle this)
  → app.invoke(None, config) to resume
  → ToolNode executes create_jira_ticket
  → final answer returned to API
  → webhook/notification sent to original requester
```

**Database record:**
```sql
INSERT INTO pending_approvals (thread_id, user_id, pending_tools, created_at, status)
VALUES ($1, $2, $3, NOW(), 'pending');
```

This table powers the approval UI listing. When approved/rejected, update status and call the resume endpoint.

**Timeout handling:**
```python
# Cron job: expire approvals older than 24h
UPDATE pending_approvals SET status = 'expired'
WHERE status = 'pending' AND created_at < NOW() - INTERVAL '24 hours';
```

---

## Q4. How would you scale this system to handle 10,000 concurrent users?

**Answer:**
Each component scales independently:

**FastAPI layer:**
- Horizontal: run 20-50 stateless async instances behind an ALB
- Each handles 200-500 concurrent SSE connections
- SSE sticky routing: consistent hash on `thread_id` at the LB level

**PostgreSQL:**
- PgBouncer connection pooler: 500 app connections → 100 DB connections
- Read replicas for `get_state` / `get_state_history` (read-heavy)
- Checkpoint writes go to primary only
- Partition `checkpoints` table by `thread_id` hash (even write distribution)

**Redis:**
- ElastiCache cluster mode: shard by key prefix
- Rate limit keys auto-expire; no unbounded growth

**LLM API:**
- 10K users × avg 500 tokens/request = 5M tokens/min
- OpenAI Tier 3 is 2M TPM for gpt-4o-mini — need an enterprise agreement or batching
- Use LLM router: gpt-4o-mini for simple queries, downgrade to GPT-3.5 for non-LLM-safety-critical paths

**Bottleneck:** the LLM API rate limit, not your infrastructure. At scale, request queuing (Celery + Redis) and streaming progressive results become essential.

---

## Q5. How do you handle the case where a user's message triggers 15 tool calls and the context window fills up?

**Answer:**
This is the "context overflow" problem. Strategies in order of preference:

**1. Summarise old messages (context compression):**
```python
def compress_if_needed(state: State) -> State:
    messages  = state["messages"]
    total_tok = sum_tokens(messages)  # estimate with tiktoken
    if total_tok < 80_000:
        return state

    # Keep system message + last 10 messages
    old   = messages[:-10]
    kept  = messages[-10:]
    summary = llm.invoke([
        SystemMessage("Summarise these messages in 200 words."),
        *old,
    ])
    return {"messages": [SystemMessage(f"[Summary of earlier conversation]: {summary.content}"), *kept]}
```

**2. Cap tool outputs:**
```python
@tool
def get_logs(service: str) -> str:
    """Get last 50 log lines for a service."""
    logs = fetch_logs(service, lines=10000)
    return "\n".join(logs[:50])  # never return unbounded output
```

**3. Use a longer-context model:**
- gpt-4o: 128K context — 4× more room than gpt-4o-mini
- Claude claude-opus-4-8: 200K context — useful for very long conversations

**4. Limit recursion:**
```python
app = builder.compile(recursion_limit=25)  # default is 25
```
This raises `GraphRecursionError` after 25 node executions, preventing infinite loops.

---

## Q6. How would you add A/B testing to compare two versions of the agent prompt?

**Answer:**

**Approach: traffic split at the request layer:**
```python
import random

def select_prompt_variant(user_id: str) -> str:
    # Consistent assignment per user (same user always gets same variant)
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "variant_b" if hash_val % 100 < 20 else "control"  # 20% B

@api.post("/chat")
async def chat(req: ChatRequest, user_id: str = Depends(get_user_id)):
    variant = select_prompt_variant(user_id)
    config  = {
        "configurable": {"thread_id": req.thread_id},
        "metadata": {"user_id": user_id, "ab_variant": variant},
        "tags":     [variant],
    }
    # Select the right compiled graph
    graph = agent_variants[variant]
    result = await graph.ainvoke(...)
```

**Measurement via LangSmith:**
- Filter runs by tag `"control"` vs `"variant_b"`
- Compare: user ratings (thumbs up), tool call accuracy, latency, token cost
- After N samples (typically 1000+), run statistical significance test

**Guardrail:** never A/B test safety-critical paths. Only test quality improvements (prompt wording, temperature, routing thresholds).

---

## Q7. How would you prevent runaway agent loops from running up the OpenAI bill?

**Answer:**
Multiple layers of protection:

**1. `recursion_limit` (LangGraph):**
```python
app = builder.compile(recursion_limit=15)  # GraphRecursionError after 15 nodes
```

**2. `max_tokens` (per LLM call):**
```python
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000)
```

**3. Per-user daily spend cap (Redis):**
```python
DAILY_CAP_USD = 2.00  # $2/user/day

async def check_daily_cap(user_id: str, tokens: int) -> None:
    key  = f"spend:{user_id}:{date.today()}"
    cost = tokens * 0.15 / 1_000_000   # $0.15 per 1M tokens
    spent = float(await redis.incrbyfloat(key, cost))
    if spent == cost:
        await redis.expire(key, 86400)
    if spent > DAILY_CAP_USD:
        raise HTTPException(402, "Daily usage limit reached.")
```

**4. Request timeout:**
```python
llm = ChatOpenAI(model="gpt-4o-mini", timeout=30)
# If any single LLM call takes > 30s, raise TimeoutError
```

**5. Monitoring alert:**
```
Alert: if any single thread_id generates > 100 tool calls in 10 minutes
→ page on-call, suspend that thread
```

---

## Q8. How do you handle OpenAI API downtime in a production system?

**Answer:**
OpenAI has had several outages. A resilient system handles this at multiple levels:

**1. Exponential backoff on transient errors (429, 503):**
```python
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

@retry(
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIStatusError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
async def call_llm(messages):
    return await llm.ainvoke(messages)
```

**2. Fallback model:**
```python
async def call_with_fallback(messages):
    try:
        return await primary_llm.ainvoke(messages)   # gpt-4o-mini
    except openai.APIStatusError:
        return await fallback_llm.ainvoke(messages)  # Claude claude-haiku-4-5-20251001 via Anthropic
```

**3. Request queuing:**
For non-streaming requests, queue them in Redis. A worker processes them when the API recovers. Return `202 Accepted` immediately with a job ID; client polls `/job/{id}` for the result.

**4. Graceful degradation:**
For the tools that don't require LLM (exact keyword search, status checks), let those work even during LLM outage.

**5. Status page monitoring:**
Subscribe to OpenAI's status page RSS feed; alert on incidents via PagerDuty before users report problems.
