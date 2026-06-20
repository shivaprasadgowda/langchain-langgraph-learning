"""
Concept: FastAPI + LangGraph — the production API layer.

This file shows the full FastAPI application that wraps a LangGraph agent:
  POST /chat          — invoke (non-streaming, returns final answer)
  GET  /chat/stream   — SSE streaming via astream_events
  POST /chat/approve  — resume a human-in-the-loop interrupt
  GET  /chat/history  — return thread conversation history
  GET  /health        — liveness probe for load balancer

This is a runnable sketch. It prints the full code with explanation.
To actually run it:
    pip install fastapi uvicorn sse-starlette
    uvicorn 15_production_architecture.02_fastapi_langgraph:api --reload

Run (explanation):
    python 15_production_architecture/02_fastapi_langgraph.py
"""


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


section("Full FastAPI application code")
print('''
  # ── Imports ──────────────────────────────────────────────────────────────────

  import json
  import os
  from contextlib import asynccontextmanager
  from typing import Annotated, AsyncGenerator

  from dotenv import load_dotenv
  from fastapi import Depends, FastAPI, HTTPException, Header, Query
  from fastapi.responses import StreamingResponse
  from pydantic import BaseModel
  from langchain_core.messages import HumanMessage

  # LangGraph imports
  from langgraph.graph import StateGraph, START, END
  from langgraph.graph.message import add_messages
  from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
  from psycopg_pool import AsyncConnectionPool

  load_dotenv()


  # ── 1. Database lifespan (connection pool) ────────────────────────────────────

  pool: AsyncConnectionPool | None = None
  app_graph = None                          # compiled LangGraph app

  @asynccontextmanager
  async def lifespan(api: FastAPI):
      """Open DB pool and compile graph on startup; close pool on shutdown."""
      global pool, app_graph

      pool = AsyncConnectionPool(
          conninfo=os.environ["POSTGRES_URL"],
          min_size=2,
          max_size=10,
          open=False,
      )
      await pool.open()

      checkpointer = AsyncPostgresSaver(pool)
      await checkpointer.setup()            # creates checkpoint tables if needed

      app_graph = build_graph(checkpointer) # your compiled StateGraph

      yield                                 # app runs here

      await pool.close()


  api = FastAPI(title="DevOps Assistant", lifespan=lifespan)


  # ── 2. Request / response models ──────────────────────────────────────────────

  class ChatRequest(BaseModel):
      message:   str
      thread_id: str

  class ChatResponse(BaseModel):
      answer:    str
      thread_id: str

  class ApproveRequest(BaseModel):
      thread_id: str
      approved:  bool
      reason:    str | None = None


  # ── 3. Auth dependency ────────────────────────────────────────────────────────

  async def get_user_id(authorization: Annotated[str, Header()]) -> str:
      """Verify JWT and return user_id. Raises 401 on invalid token."""
      if not authorization.startswith("Bearer "):
          raise HTTPException(status_code=401, detail="Missing Bearer token")
      token = authorization.removeprefix("Bearer ")
      # In production: verify with python-jose or PyJWT
      #   payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
      #   return payload["sub"]
      # For demo, accept any non-empty token and use it as user_id
      if not token:
          raise HTTPException(status_code=401, detail="Invalid token")
      return f"user-{token[:8]}"


  # ── 4. POST /chat — non-streaming invoke ──────────────────────────────────────

  @api.post("/chat", response_model=ChatResponse)
  async def chat(
      req:     ChatRequest,
      user_id: str = Depends(get_user_id),
  ) -> ChatResponse:
      config = {
          "configurable": {"thread_id": req.thread_id},
          "metadata":     {"user_id": user_id},
          "tags":         ["production"],
      }
      result = await app_graph.ainvoke(
          {"messages": [HumanMessage(content=req.message)]},
          config=config,
      )
      return ChatResponse(
          answer    = result["messages"][-1].content,
          thread_id = req.thread_id,
      )


  # ── 5. GET /chat/stream — SSE streaming ───────────────────────────────────────

  async def event_generator(
      message:   str,
      thread_id: str,
      user_id:   str,
  ) -> AsyncGenerator[str, None]:
      config = {
          "configurable": {"thread_id": thread_id},
          "metadata":     {"user_id": user_id},
          "tags":         ["production", "stream"],
      }
      try:
          async for event in app_graph.astream_events(
              {"messages": [HumanMessage(content=message)]},
              config=config,
              version="v2",
          ):
              ev   = event["event"]
              node = event.get("metadata", {}).get("langgraph_node", "")

              if ev == "on_chat_model_stream":
                  chunk = event["data"]["chunk"]
                  if chunk.content and not chunk.tool_call_chunks:
                      yield f"data: {json.dumps({'type': 'token', 'text': chunk.content})}\\n\\n"

              elif ev == "on_tool_start" and node == "tools":
                  yield f"data: {json.dumps({'type': 'tool_start', 'name': event['name']})}\\n\\n"

              elif ev == "on_tool_end" and node == "tools":
                  out = event["data"].get("output", "")
                  content = out.content if hasattr(out, "content") else str(out)
                  yield f"data: {json.dumps({'type': 'tool_end', 'result': content[:200]})}\\n\\n"

      except Exception as e:
          yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\\n\\n"

      yield f"data: {json.dumps({'type': 'done'})}\\n\\n"


  @api.get("/chat/stream")
  async def chat_stream(
      message:   str = Query(...),
      thread_id: str = Query(...),
      token:     str = Query(...),   # EventSource can\'t send headers; pass as query param
  ) -> StreamingResponse:
      user_id = await get_user_id(f"Bearer {token}")
      return StreamingResponse(
          event_generator(message, thread_id, user_id),
          media_type="text/event-stream",
          headers={
              "Cache-Control":               "no-cache",
              "X-Accel-Buffering":           "no",
              "Access-Control-Allow-Origin": "*",
          },
      )


  # ── 6. POST /chat/approve — resume HITL interrupt ─────────────────────────────

  @api.post("/chat/approve")
  async def approve(
      req:     ApproveRequest,
      user_id: str = Depends(get_user_id),
  ) -> ChatResponse:
      config = {"configurable": {"thread_id": req.thread_id}}

      if req.approved:
          # Resume: pass None as input — graph continues from interrupt point
          result = await app_graph.ainvoke(None, config=config)
      else:
          # Reject: update state so the graph returns without executing the tool
          await app_graph.aupdate_state(
              config,
              {"messages": [{"role": "assistant", "content": f"Action cancelled: {req.reason}"}]},
          )
          result = await app_graph.aupdate_state(config, {})  # finalize

      return ChatResponse(
          answer    = result["messages"][-1].content,
          thread_id = req.thread_id,
      )


  # ── 7. GET /chat/history — thread history ──────────────────────────────────────

  @api.get("/chat/history")
  async def history(
      thread_id: str = Query(...),
      user_id:   str = Depends(get_user_id),
  ) -> dict:
      config    = {"configurable": {"thread_id": thread_id}}
      state     = await app_graph.aget_state(config)
      messages  = state.values.get("messages", [])
      return {
          "thread_id": thread_id,
          "messages":  [
              {"role": m.type, "content": m.content}
              for m in messages
              if hasattr(m, "content") and m.content
          ],
      }


  # ── 8. GET /health — liveness probe ───────────────────────────────────────────

  @api.get("/health")
  async def health() -> dict:
      # Check DB connectivity
      async with pool.connection() as conn:
          await conn.execute("SELECT 1")
      return {"status": "ok", "db": "connected"}
''')


section("Key design decisions")
print("""
  1. async everywhere
     Use ainvoke() / astream_events() — NOT the sync versions.
     Sync LLM calls block the event loop and starve all other requests.

  2. lifespan for connection pool
     Open the DB pool once on startup; share it across all requests.
     Never create a new pool per request — that's a connection leak.

  3. EventSource auth via query param
     Browser EventSource cannot set headers. Pass a short-lived token
     as a query param for the SSE endpoint. Generate it via:
       POST /auth/streaming-token → {"token": "...", "expires_in": 30}

  4. thread_id owned by the client
     The frontend generates a UUID for each conversation and passes it
     on every request. The server never assigns thread_ids — this lets
     the client maintain context without a session lookup.

  5. config metadata for observability
     Every invoke() passes {"metadata": {"user_id": ...}} so LangSmith
     runs are searchable by user. Costs can then be attributed per user.

  6. Error SSE event
     Wrap the event generator in try/except and emit
     {"type": "error", "detail": "..."} so the frontend knows the stream
     failed. Without this, EventSource just closes silently.
""")
