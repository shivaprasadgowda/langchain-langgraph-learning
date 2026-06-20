# 16 — Interview Notes

> Status: **Complete**

## What This Section Covers

Consolidated interview Q&A across the entire stack — 48 questions total (8 per topic).

## Files

| File | Topic | Questions |
|------|-------|-----------|
| `01_langchain_questions.md` | LangChain, LCEL, messages, prompt templates | 8 |
| `02_langgraph_questions.md` | StateGraph, nodes, edges, reducers, HITL, streaming | 8 |
| `03_langsmith_questions.md` | Tracing, datasets, evaluators, CI/CD, debugging | 8 |
| `04_rag_questions.md` | Indexing pipeline, retrieval strategies, citations, failure modes | 8 |
| `05_tool_calling_questions.md` | `@tool`, `bind_tools`, parallel calls, ToolNode, HITL approval | 8 |
| `06_system_design_questions.md` | Multi-tenant design, self-corrective RAG, scaling, A/B testing, outage handling | 8 |

## How to Study

Read each file top to bottom. For concepts you're not sure about, go back to the relevant section:

| Topic | Section to review |
|-------|------------------|
| LangChain basics | `01_langchain_basics/`, `02_messages_and_prompts/`, `03_structured_output/` |
| LangGraph | `06_langgraph_basics/`, `07_langgraph_router/`, `08_persistence_checkpointer/` |
| Tool calling | `04_tool_calling/`, `05_manual_agent_loop/`, `10_toolnode_agents/` |
| RAG | `09_rag/` |
| HITL | `11_human_in_the_loop/` |
| Multi-agent | `12_multi_agent_systems/` |
| Streaming | `13_streaming/` |
| LangSmith | `14_langsmith_observability/` |
| Production | `15_production_architecture/` |

## Quick Reference: The Most Common Interview Questions

**"What is LangGraph?"**
> A library for building stateful, cyclical AI agent graphs with built-in persistence, HITL, and streaming. Solves what LCEL can't: loops, checkpointing, multi-agent composition.

**"What is RAG?"**
> Retrieval-Augmented Generation — embed documents into vectors, retrieve relevant chunks at query time, and pass them as context to the LLM. Gives the LLM access to private/recent knowledge without fine-tuning.

**"How does tool calling work?"**
> LLM returns `tool_calls` (name + args + id) instead of text. You execute the function and return a `ToolMessage(content=result, tool_call_id=id)`. The LLM uses the result to generate its final answer.

**"How do you persist multi-turn conversations?"**
> `MemorySaver` for local dev. `AsyncPostgresSaver` for production — writes a checkpoint after every graph node; any server instance can resume any thread.

**"How do you stream responses?"**
> `app.stream(input, stream_mode="messages")` for token chunks. `app.astream_events(input, version="v2")` for full event stream including tool start/end. FastAPI `StreamingResponse` with SSE format for the browser.

**"Design an LLM chat system."**
> React → NGINX → FastAPI (async, JWT auth, Redis rate limit, guardrails) → LangGraph (ToolNode agent) → PostgreSQL (AsyncPostgresSaver) + OpenAI API + LangSmith tracing.
