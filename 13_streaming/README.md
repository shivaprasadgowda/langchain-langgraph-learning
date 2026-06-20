# 13 — Streaming

> Status: **Complete**

## What This Section Covers

- `model.stream` — token-level streaming
- `graph.stream` — node-level streaming
- Streaming events API
- FastAPI Server-Sent Events (SSE) concept

## Files

| File | Purpose |
|------|---------|
| `01_model_stream.py` | `llm.stream()`; chunk structure; `AIMessageChunk +`; chain streaming; collecting chunks |
| `02_graph_stream.py` | `app.stream()` with `"updates"`, `"values"`, `"messages"`, and combined modes |
| `03_streaming_events.py` | Async `astream_events()` — all events, token filter, tool filter, node lifecycle, UI event stream |
| `04_fastapi_sse_concept.py` | SSE wire format, FastAPI endpoint, React `EventSource`, production issues — no API key needed |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

```bash
python 13_streaming/01_model_stream.py
python 13_streaming/02_graph_stream.py
python 13_streaming/03_streaming_events.py   # async — uses asyncio.run()
python 13_streaming/04_fastapi_sse_concept.py  # no API key needed
```

## Streaming API Comparison

| API | Granularity | Sync? | Best for |
|-----|-------------|-------|---------|
| `llm.stream(messages)` | Token chunks | Sync | Simple scripts, standalone model calls |
| `app.stream(input, stream_mode="updates")` | Node outputs | Sync | Logging which nodes ran and what changed |
| `app.stream(input, stream_mode="messages")` | Token chunks + node info | Sync | Token streaming with node attribution |
| `app.astream_events(input, version="v2")` | All events | Async | Production UIs, tool indicators, full observability |

## Key Concepts

**`AIMessageChunk` + operator** — chunks accumulate into a full message; `chunk1 + chunk2` merges content and metadata.

**`stream_mode="messages"`** — yields `(chunk, metadata)` tuples; `metadata["langgraph_node"]` identifies which node emitted the token.

**`astream_events` event types** — `on_chat_model_stream` (tokens), `on_tool_start/end` (tools), `on_chain_start/end` (nodes).

**SSE** — unidirectional HTTP streaming; `text/event-stream` content type; `EventSource` in browser; preferred over WebSockets for LLM chat.

**Production SSE issues** — disable nginx buffering (`X-Accel-Buffering: no`), keep-alive comments every 15s, rate-limit concurrent connections.
