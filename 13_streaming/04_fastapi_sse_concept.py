"""
Concept: FastAPI + Server-Sent Events (SSE) for streaming LangGraph output.

Server-Sent Events (SSE) is the standard web protocol for server→client
streaming. The server sends a stream of text/event-stream responses;
the browser's EventSource API receives them in real time.

This file is a code sketch with detailed comments.
It does NOT start a server — it prints the patterns with explanation.

To actually run the FastAPI server (after installing dependencies):
    pip install fastapi uvicorn sse-starlette
    uvicorn 13_streaming.04_fastapi_sse_concept:api --reload

Run (explanation only):
    python 13_streaming/04_fastapi_sse_concept.py
"""

import json


def section(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


# ═══════════════════════════════════════════════════════════════════════════════
section("1. The SSE wire format")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  SSE is plain text over HTTP. Each event looks like:

    data: {"type": "token", "text": "The weather"}\\n
    data: {"type": "token", "text": " in Tokyo"}\\n
    data: {"type": "tool_start", "name": "get_weather"}\\n
    data: {"type": "token", "text": " is 22°C."}\\n
    data: {"type": "done"}\\n
    \\n   ← blank line signals end of event

  The browser's EventSource API receives these automatically:
    const es = new EventSource('/chat/stream?thread_id=abc');
    es.onmessage = (e) => {
        const event = JSON.parse(e.data);
        if (event.type === 'token') appendToUI(event.text);
        if (event.type === 'done')  es.close();
    };
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("2. Full FastAPI SSE endpoint")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  from fastapi import FastAPI
  from fastapi.responses import StreamingResponse
  from langchain_core.messages import HumanMessage
  from langgraph.graph import StateGraph  # your compiled graph

  api = FastAPI()

  async def event_generator(message: str, thread_id: str):
      \"\"\"Async generator that yields SSE-formatted lines.\"\"\"
      config = {"configurable": {"thread_id": thread_id}}

      async for event in app.astream_events(
          {"messages": [HumanMessage(content=message)]},
          config=config,
          version="v2",
      ):
          ev   = event["event"]
          node = event.get("metadata", {}).get("langgraph_node", "")

          # LLM token chunk
          if ev == "on_chat_model_stream":
              chunk = event["data"]["chunk"]
              if chunk.content and not chunk.tool_call_chunks:
                  payload = json.dumps({"type": "token", "text": chunk.content})
                  yield f"data: {payload}\\n\\n"

          # Tool starting
          elif ev == "on_tool_start" and node == "tools":
              payload = json.dumps({"type": "tool_start", "name": event["name"]})
              yield f"data: {payload}\\n\\n"

          # Tool finished
          elif ev == "on_tool_end" and node == "tools":
              out = event["data"].get("output", "")
              content = out.content if hasattr(out, "content") else str(out)
              payload = json.dumps({"type": "tool_end", "result": content[:200]})
              yield f"data: {payload}\\n\\n"

      # Signal completion
      yield f"data: {json.dumps({'type': 'done'})}\\n\\n"


  @api.get("/chat/stream")
  async def chat_stream(message: str, thread_id: str):
      return StreamingResponse(
          event_generator(message, thread_id),
          media_type="text/event-stream",
          headers={
              "Cache-Control":               "no-cache",
              "X-Accel-Buffering":           "no",   # disable nginx buffering
              "Access-Control-Allow-Origin": "*",     # CORS for the browser
          },
      )
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("3. sse-starlette alternative (cleaner)")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  pip install sse-starlette

  from sse_starlette.sse import EventSourceResponse

  @api.get("/chat/stream")
  async def chat_stream(message: str, thread_id: str):
      async def generator():
          async for event in app.astream_events(...):
              ...
              yield {"data": json.dumps(payload)}   # sse-starlette handles framing

      return EventSourceResponse(generator())

  EventSourceResponse handles the text/event-stream framing automatically,
  including keep-alive pings every 15 seconds to prevent proxy timeouts.
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("4. React frontend (EventSource)")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  // React component (TypeScript)
  function ChatMessage({ message, threadId }) {
    const [text, setText] = useState("");
    const [toolStatus, setToolStatus] = useState(null);

    function sendMessage() {
      const url = `/chat/stream?message=${encodeURIComponent(message)}&thread_id=${threadId}`;
      const es  = new EventSource(url);

      es.onmessage = (e) => {
        const event = JSON.parse(e.data);

        switch (event.type) {
          case "token":
            setText(prev => prev + event.text);   // stream tokens into UI
            break;
          case "tool_start":
            setToolStatus(`🔧 Calling ${event.name}...`);
            break;
          case "tool_end":
            setToolStatus(null);
            break;
          case "done":
            es.close();
            break;
        }
      };

      es.onerror = () => es.close();
    }
    ...
  }
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("5. Production considerations")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  NGINX / proxy buffering:
    Set "X-Accel-Buffering: no" to prevent nginx from buffering SSE responses.
    Without it, tokens are batched and sent in bursts rather than streaming.

  Connection timeout:
    Load balancers often close idle connections after 60s. If the LLM is slow,
    send a keep-alive comment every 15s:
        yield ": keep-alive\\n\\n"   # SSE comment, browser ignores it

  Reconnection:
    EventSource reconnects automatically on disconnect. Use the "id:" SSE field
    and "Last-Event-ID" header to resume streaming from the last event ID.

  Authentication:
    EventSource does not support custom headers. Pass auth via:
    - Query parameter: /stream?token=...  (less secure — appears in logs)
    - Cookie: set an HttpOnly cookie before the SSE request
    - POST to get a short-lived streaming token, then GET /stream?token=...

  Rate limiting:
    Each SSE connection holds an open HTTP connection. Limit concurrent
    streaming sessions per user to prevent resource exhaustion.
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("6. What the full stack looks like")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  Browser (React)
       │  GET /chat/stream?message=...&thread_id=...
       │  EventSource
       ▼
  FastAPI (Python)
       │  astream_events() on compiled LangGraph app
       │  Yields SSE events: token / tool_start / tool_end / done
       ▼
  LangGraph (compiled StateGraph)
       │  LLM node → ToolNode → LLM node → ...
       ▼
  OpenAI API  +  Tool execution (Jira / AWS / K8s APIs)
       │
  PostgreSQL (AsyncPostgresSaver)
       └─ State checkpoint written after every node
""")
