# LangChain, LangGraph & LangSmith Learning Reference

A personal learning and interview-preparation reference covering LangChain, LangGraph, LangSmith, RAG, tool calling, agentic workflows, and production AI chatbot architecture.

## Goal

Build deep, practical understanding of the modern AI agent stack — from a first LLM call all the way to production-grade multi-agent systems.

## Sections

| # | Folder | Topic |
|---|--------|-------|
| 00 | `00_setup/` | Environment setup, packages, `.env` |
| 01 | `01_langchain_basics/` | ChatOpenAI, first LLM call |
| 02 | `02_messages_and_prompts/` | Messages, PromptTemplates, LCEL |
| 03 | `03_structured_output/` | Pydantic, `with_structured_output` |
| 04 | `04_tool_calling/` | `@tool`, `bind_tools`, tool calls |
| 05 | `05_manual_agent_loop/` | ToolMessage, manual agent loop |
| 06 | `06_langgraph_basics/` | StateGraph, nodes, edges |
| 07 | `07_langgraph_router/` | Classifier, conditional edges |
| 08 | `08_persistence_checkpointer/` | InMemorySaver, thread IDs |
| 09 | `09_rag/` | Document loading, embeddings, retrieval |
| 10 | `10_toolnode_agents/` | ToolNode, tools_condition |
| 11 | `11_human_in_the_loop/` | Interrupt/resume, approval flows |
| 12 | `12_multi_agent_systems/` | Supervisor + specialist agents |
| 13 | `13_streaming/` | model.stream, graph.stream, SSE |
| 14 | `14_langsmith_observability/` | Tracing, debugging, evaluation |
| 15 | `15_production_architecture/` | FastAPI, PostgreSQL, Redis, auth |
| 16 | `16_interview_notes/` | Interview Q&A across all topics |

## Quick Start

```bash
cd 00_setup
python -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
cp ../.env.example ../.env
# Fill in your API keys in .env, then:
python 01_verify_setup.py
```

## Prerequisites

- Python 3.11+
- OpenAI API key
- (Optional) LangSmith API key for tracing
