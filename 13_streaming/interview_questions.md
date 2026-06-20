# Interview Questions — 13 Streaming

---

## Q1. Why does streaming matter for chat UX and what latency improvement does it provide?

**Answer:**
Without streaming, the user sees nothing until the model finishes generating — for a 200-token response at 50 tokens/sec that is a 4-second blank screen. With streaming, the first token appears after ~80-200ms (time-to-first-byte), and the user reads along as the model writes.

Perceived responsiveness is much more important than raw throughput for interactive applications. Studies consistently show users prefer a slow stream they can read over a fast result that arrives all at once after a delay.

---

## Q2. What is the difference between `llm.stream()`, `app.stream()`, and `app.astream_events()`?

**Answer:**

| API | Granularity | Sync/Async | What it yields |
|-----|-------------|-----------|----------------|
| `llm.stream(messages)` | Token chunks | Sync | `AIMessageChunk` — one per token |
| `app.stream(input)` | Node outputs | Sync | `{node_name: state_update}` after each node |
| `app.astream_events(input)` | Everything | Async | Individual events: tokens, tool start/end, node start/end |

`llm.stream` is used inside a node when building a custom streaming node. `app.stream` is used by the caller to see which nodes ran and what they produced. `astream_events` is used by production UIs that need fine-grained visibility into all activity.

---

## Q3. What are the three `stream_mode` options for `app.stream()` and when do you use each?

**Answer:**

| Mode | Yields | Use case |
|------|--------|---------|
| `"updates"` (default) | `{node_name: partial_state_update}` — only what changed | Logging, debugging, progress indicators |
| `"values"` | Full accumulated state after each node | When downstream code needs complete state at each step |
| `"messages"` | `(AIMessageChunk, metadata)` tuples | Token-level streaming with node attribution |

`"updates"` is the lightest and most common. `"messages"` is what you use to stream tokens from within a graph to a UI. You can combine modes: `stream_mode=["updates", "messages"]`.

---

## Q4. What event types does `astream_events()` emit and what does each contain?

**Answer:**

| Event | Trigger | Key data field |
|-------|---------|---------------|
| `on_chat_model_stream` | Each LLM token chunk | `data["chunk"]` — `AIMessageChunk` |
| `on_tool_start` | Tool begins executing | `data["input"]` — tool arguments |
| `on_tool_end` | Tool finishes | `data["output"]` — tool result |
| `on_chain_start` | Node / chain begins | `metadata["langgraph_node"]` — node name |
| `on_chain_end` | Node / chain finishes | `data["output"]` — node output |
| `on_retriever_start/end` | Retriever called (RAG) | `data["query"]` / `data["documents"]` |

Every event also has `run_id` for correlation, `tags` for filtering, and `metadata["langgraph_node"]` for identifying which graph node emitted it.

---

## Q5. How do you stream LLM tokens from inside a ToolNode agent graph?

**Answer:**
Use `stream_mode="messages"` on `app.stream()`:

```python
for chunk, metadata in app.stream(input, stream_mode="messages"):
    node = metadata.get("langgraph_node", "")
    if chunk.type == "AIMessageChunk" and chunk.content:
        # Filter out tool-call generation chunks (no text content)
        if not chunk.tool_call_chunks:
            print(chunk.content, end="", flush=True)
```

Or use `astream_events` and filter for `on_chat_model_stream`:

```python
async for event in app.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk.content and not chunk.tool_call_chunks:
            yield chunk.content  # send to UI
```

---

## Q6. What is SSE and why is it preferred over WebSockets for LLM streaming?

**Answer:**
SSE (Server-Sent Events) is a unidirectional HTTP streaming protocol — the server pushes events to the client over a single long-lived connection. The browser's `EventSource` API handles it natively.

For LLM chat:
- **SSE is simpler** — no handshake, no full-duplex setup, works through HTTP/2 proxies.
- **One-directional is sufficient** — the server streams tokens; the client sends a new HTTP request for each user message (not a persistent bidirectional channel).
- **Better proxy compatibility** — WebSockets require explicit proxy support; SSE is plain HTTP.

WebSockets are better for truly bidirectional real-time applications (collaborative editing, gaming, live cursors).

---

## Q7. What production issues must you handle for SSE endpoints?

**Answer:**

| Issue | Fix |
|-------|-----|
| Nginx buffering tokens in batches | Set `X-Accel-Buffering: no` header |
| Load balancer closing idle connections | Send `: keep-alive\n\n` comment every 15s |
| `EventSource` cannot send auth headers | Use HttpOnly cookie or short-lived token in query param |
| Too many open connections | Rate-limit concurrent SSE sessions per user |
| Reconnect on disconnect | Use `id:` SSE field; client sends `Last-Event-ID` header on reconnect |

---

## Q8. How does `astream_events` help build "thinking indicator" UI for tool calls?

**Answer:**
Filter for `on_tool_start` and `on_tool_end` events:

```python
async for event in app.astream_events(input, version="v2"):
    if event["event"] == "on_tool_start":
        tool_name = event["name"]
        # Emit to UI: show "🔧 Searching Jira..." spinner
        yield {"type": "tool_start", "name": tool_name}

    elif event["event"] == "on_tool_end":
        # Emit to UI: hide spinner
        yield {"type": "tool_end", "name": event["name"]}

    elif event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk.content and not chunk.tool_call_chunks:
            # Emit to UI: stream token into text box
            yield {"type": "token", "text": chunk.content}
```

The frontend renders a "🔧 Calling get_aws_cost..." indicator between `tool_start` and `tool_end`, then streams the final answer token by token. This is the pattern used by ChatGPT's "searching" and Claude's tool-use indicators.
