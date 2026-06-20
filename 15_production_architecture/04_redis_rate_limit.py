"""
Concept: Rate limiting, auth, and guardrails for production LLM APIs.

Three defences every production LLM endpoint needs:

  1. Rate limiting (Redis)
     Prevents one user from exhausting your OpenAI quota or running up cost.
     Pattern: sliding window counter in Redis (INCR + EXPIRE).

  2. Auth (JWT)
     Identifies who is making requests so you can enforce per-user limits,
     attribute costs, audit usage, and ban bad actors.

  3. Guardrails (input + output)
     Input:  block harmful prompts before they reach the LLM (saves cost too).
     Output: redact PII and block harmful content before returning to client.

Run:
    python 15_production_architecture/04_redis_rate_limit.py
"""


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


# ═══════════════════════════════════════════════════════════════════════════════
section("1. Redis sliding-window rate limiter")
# ═══════════════════════════════════════════════════════════════════════════════
print('''
  import redis.asyncio as redis
  from fastapi import HTTPException, Request

  redis_client = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)

  RATE_LIMIT_REQUESTS = 20      # max requests
  RATE_LIMIT_WINDOW   = 60      # per 60 seconds
  COST_LIMIT_CENTS    = 100     # max $1.00 per user per day


  async def check_rate_limit(user_id: str) -> None:
      """Raises 429 if user has exceeded the request rate limit."""
      key     = f"rate:{user_id}"
      count   = await redis_client.incr(key)

      if count == 1:
          # First request in this window — set expiry
          await redis_client.expire(key, RATE_LIMIT_WINDOW)

      if count > RATE_LIMIT_REQUESTS:
          ttl = await redis_client.ttl(key)
          raise HTTPException(
              status_code=429,
              detail=f"Rate limit exceeded. Try again in {ttl}s.",
              headers={"Retry-After": str(ttl)},
          )


  async def check_daily_cost(user_id: str, estimated_tokens: int) -> None:
      """Raises 402 if user would exceed daily cost cap."""
      cost_key     = f"cost:{user_id}:{date.today()}"
      # cost in micro-cents (integer arithmetic, no float rounding)
      token_cost   = estimated_tokens * 15    # 15 µ¢ per token (gpt-4o-mini input)
      daily_spend  = await redis_client.incrby(cost_key, token_cost)

      if daily_spend == token_cost:             # first request today
          await redis_client.expire(cost_key, 86400)

      if daily_spend > COST_LIMIT_CENTS * 10000:
          raise HTTPException(status_code=402, detail="Daily cost limit reached.")


  # FastAPI dependency — add to any endpoint
  async def rate_limit(request: Request, user_id: str = Depends(get_user_id)):
      await check_rate_limit(user_id)
      return user_id

  # Usage:
  @api.post("/chat")
  async def chat(req: ChatRequest, user_id: str = Depends(rate_limit)):
      ...
''')


# ═══════════════════════════════════════════════════════════════════════════════
section("2. JWT authentication")
# ═══════════════════════════════════════════════════════════════════════════════
print('''
  # Install: pip install python-jose[cryptography]
  from jose import JWTError, jwt
  from fastapi import HTTPException, Header
  from typing import Annotated

  JWT_SECRET    = os.environ["JWT_SECRET"]    # 32-byte random secret
  JWT_ALGORITHM = "HS256"
  JWT_EXPIRE_M  = 60 * 24   # 24 hours


  def create_token(user_id: str) -> str:
      """Issued at login."""
      payload = {
          "sub": user_id,
          "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_M),
          "iat": datetime.utcnow(),
      }
      return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


  async def get_user_id(authorization: Annotated[str, Header()]) -> str:
      """FastAPI dependency: verify JWT and return user_id."""
      if not authorization.startswith("Bearer "):
          raise HTTPException(status_code=401, detail="Missing Bearer token")
      token = authorization.removeprefix("Bearer ")
      try:
          payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
          return payload["sub"]
      except JWTError:
          raise HTTPException(status_code=401, detail="Invalid or expired token")


  # SSE workaround: EventSource can\'t send headers.
  # Issue a short-lived (30s) streaming token via a regular POST:
  @api.post("/auth/streaming-token")
  async def streaming_token(user_id: str = Depends(get_user_id)) -> dict:
      token = create_token_with_expiry(user_id, expires_seconds=30)
      return {"token": token}

  # Then the browser opens: GET /chat/stream?token=<30-second-token>
''')


# ═══════════════════════════════════════════════════════════════════════════════
section("3. Input guardrails")
# ═══════════════════════════════════════════════════════════════════════════════
print('''
  Input guardrails run BEFORE the LangGraph agent — they save cost and
  prevent the LLM from being misused.


  ── a) Length limit ──

  MAX_MESSAGE_CHARS = 2000

  def check_input_length(message: str) -> None:
      if len(message) > MAX_MESSAGE_CHARS:
          raise HTTPException(
              status_code=400,
              detail=f"Message too long ({len(message)} chars). Max {MAX_MESSAGE_CHARS}.",
          )


  ── b) Keyword blocklist (fast, no LLM cost) ──

  BLOCKED_PATTERNS = [
      "ignore previous instructions",
      "disregard your system prompt",
      "you are now DAN",
  ]

  def check_prompt_injection(message: str) -> None:
      lower = message.lower()
      for pattern in BLOCKED_PATTERNS:
          if pattern in lower:
              raise HTTPException(
                  status_code=400,
                  detail="Message blocked by content policy.",
              )


  ── c) LLM-based content classifier (more accurate, costs ~$0.0001) ──

  from langchain_openai import ChatOpenAI
  from pydantic import BaseModel

  class SafetyCheck(BaseModel):
      safe:   bool
      reason: str

  safety_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
  safety_chain = safety_llm.with_structured_output(SafetyCheck)

  async def check_content_safety(message: str) -> None:
      result: SafetyCheck = await safety_chain.ainvoke([
          ("system", "Classify if this message is safe for a DevOps work assistant. "
                     "Unsafe = harmful, illegal, or unrelated personal content."),
          ("human", message),
      ])
      if not result.safe:
          raise HTTPException(
              status_code=400,
              detail=f"Message blocked: {result.reason}",
          )
''')


# ═══════════════════════════════════════════════════════════════════════════════
section("4. Output guardrails")
# ═══════════════════════════════════════════════════════════════════════════════
print(r'''
  Output guardrails run AFTER the LangGraph agent returns — they prevent
  sensitive data from leaking to the client.


  ── a) PII redaction (regex-based, zero latency) ──

  import re

  PII_PATTERNS = {
      "email":       re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
      "phone_us":    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
      "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
      "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
      "aws_key":     re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
  }

  def redact_pii(text: str) -> str:
      for label, pattern in PII_PATTERNS.items():
          text = pattern.sub(f"[{label.upper()} REDACTED]", text)
      return text

  # Usage after graph returns:
  answer = result["messages"][-1].content
  safe_answer = redact_pii(answer)


  ── b) Max output length ──

  MAX_RESPONSE_CHARS = 4000

  def truncate_response(text: str) -> str:
      if len(text) <= MAX_RESPONSE_CHARS:
          return text
      return text[:MAX_RESPONSE_CHARS] + "\n\n[Response truncated]"


  ── c) LLM output quality check (optional, adds latency) ──

  class OutputCheck(BaseModel):
      appropriate: bool
      reason:      str

  async def check_output_safety(response: str) -> str:
      result = await safety_chain.ainvoke([
          ("system", "Check if this AI response is appropriate for a corporate DevOps tool."),
          ("human", response),
      ])
      if not result.appropriate:
          return "I\'m sorry, I cannot provide that information."
      return response
''')


# ═══════════════════════════════════════════════════════════════════════════════
section("5. Composing all middleware in FastAPI")
# ═══════════════════════════════════════════════════════════════════════════════
print('''
  @api.post("/chat")
  async def chat(
      req:     ChatRequest,
      user_id: str = Depends(get_user_id),          # 1. Auth
  ) -> ChatResponse:
      await check_rate_limit(user_id)               # 2. Rate limit
      check_input_length(req.message)               # 3. Input: length
      check_prompt_injection(req.message)           # 4. Input: injection
      await check_content_safety(req.message)       # 5. Input: LLM safety

      config = {"configurable": {"thread_id": req.thread_id},
                "metadata": {"user_id": user_id}}

      result = await app_graph.ainvoke(
          {"messages": [HumanMessage(content=req.message)]},
          config=config,
      )

      answer = result["messages"][-1].content
      answer = redact_pii(answer)                   # 6. Output: PII
      answer = truncate_response(answer)            # 7. Output: length
      # answer = await check_output_safety(answer)  # 8. Output: safety (optional)

      return ChatResponse(answer=answer, thread_id=req.thread_id)


  # Cost of this middleware stack per request:
  #   Auth check      : < 1ms  (in-process JWT verify)
  #   Rate limit      : < 1ms  (Redis INCR, local Redis)
  #   Length check    : < 1ms  (len() in Python)
  #   Injection check : < 1ms  (string search)
  #   LLM safety      : ~200ms (extra LLM call — optional for MVP)
  #   PII redaction   : < 1ms  (regex)
  # Total overhead vs baseline: 2-5ms without LLM safety; 200ms with
''')


# ═══════════════════════════════════════════════════════════════════════════════
section("6. Redis for session caching")
# ═══════════════════════════════════════════════════════════════════════════════
print('''
  # Cache the last N messages to avoid DB reads on every streaming heartbeat
  import json

  CACHE_TTL = 300  # 5 minutes

  async def cache_thread_summary(thread_id: str, summary: str) -> None:
      await redis_client.setex(
          f"summary:{thread_id}",
          CACHE_TTL,
          summary,
      )

  async def get_cached_summary(thread_id: str) -> str | None:
      return await redis_client.get(f"summary:{thread_id}")

  # Use this to avoid re-querying PostgreSQL for read-only history requests:
  @api.get("/chat/summary")
  async def get_summary(thread_id: str, user_id: str = Depends(get_user_id)):
      cached = await get_cached_summary(thread_id)
      if cached:
          return {"summary": cached, "source": "cache"}
      # Fall through to PostgreSQL
      state   = await app_graph.aget_state({"configurable": {"thread_id": thread_id}})
      summary = state.values.get("summary", "")
      await cache_thread_summary(thread_id, summary)
      return {"summary": summary, "source": "db"}
''')
