# LangChain, LangGraph, LangSmith Learning Repository

## Project Goal

This repository is my personal learning and interview-preparation reference for:

- LangChain
- LangGraph
- LangSmith
- RAG
- Tool calling
- Agentic workflows
- Production AI chatbot architecture

The repo should be beginner-friendly, interview-focused, and practical.

## Important Working Rule

Work one section at a time.

Do not generate all sections in one response unless explicitly asked.

For every section:
1. Create a dedicated folder.
2. Add a `README.md`.
3. Add one focused Python file per concept.
4. Add comments explaining the flow.
5. Add a simple test/demo file wherever useful.
6. Add interview questions for that section.
7. Keep examples small and runnable.

## Code Rules

- Use Python.
- Use `langchain-openai` for OpenAI models.
- Use `langgraph` for graph examples.
- Use `python-dotenv`.
- Never hardcode API keys.
- Use `.env.example`.
- Keep each example focused on one concept.
- Add clear comments in code.
- Avoid unnecessary complexity.
- Prefer latest recommended LangChain/LangGraph patterns.
- Avoid deprecated patterns unless clearly marked as legacy.

## File Naming Rule

Each concept should have its own file.

Example:

```text
01_langchain_basics/
├── README.md
├── 01_first_llm_call.py
├── 02_ai_message_response.py
├── 03_chat_openai_config.py
└── interview_questions.md



## Required Sections
00_setup/
01_langchain_basics/
02_messages_and_prompts/
03_structured_output/
04_tool_calling/
05_manual_agent_loop/
06_langgraph_basics/
07_langgraph_router/
08_persistence_checkpointer/
09_rag/
10_toolnode_agents/
11_human_in_the_loop/
12_multi_agent_systems/
13_streaming/
14_langsmith_observability/
15_production_architecture/
16_interview_notes/


00_setup
Cover:
virtual environment
package installation
.env.example
requirements.txt
how to run examples

01_langchain_basics
Cover:
what is LangChain
why not just OpenAI SDK
ChatOpenAI
first LLM call
AIMessage response

02_messages_and_prompts
Cover:
SystemMessage
HumanMessage
AIMessage
manual chat history
ChatPromptTemplate
MessagesPlaceholder
LCEL chain: prompt | model | parser

03_structured_output
Cover:
Pydantic schema
with_structured_output
classification
ticket extraction
router-ready output

04_tool_calling
Cover:
@tool
bind_tools
response.tool_calls
calculator tool
dummy Jira/AWS/Bitbucket tools

05_manual_agent_loop
Cover:
ToolMessage
tool result back to LLM
manual agent loop
max_steps safety limit
why manual loops become hard

06_langgraph_basics
Cover:
what is LangGraph
StateGraph
state
node
edge
START
END
first graph

07_langgraph_router
Cover:
classifier node
structured output
conditional edges
route to Jira/AWS/Kubernetes/General nodes

08_persistence_checkpointer
Cover:
add_messages reducer
InMemorySaver
thread_id
multiple conversations
production PostgreSQL checkpointer concept

09_rag
Cover:
document loading
splitting
embeddings
vector store
retriever
RAG chain
RAG node in LangGraph
citations concept

10_toolnode_agents
Cover:
ToolNode
tools_condition
agent loop with LangGraph
difference from manual loop

11_human_in_the_loop
Cover:
interrupt/resume concept
approval before action
example: approve before Jira ticket creation

12_multi_agent_systems
Cover:
supervisor agent
specialist agents
when to use
when not to use

13_streaming
Cover:
model.stream
graph.stream
streaming events
FastAPI SSE concept

14_langsmith_observability
Cover:
LangSmith tracing
environment variables
debugging graph execution
evaluation overview

15_production_architecture
Cover:
React + FastAPI + LangGraph
PostgreSQL checkpointer
Redis rate limiting
vector DB
auth
guardrails
monitoring

16_interview_notes
Cover:
LangChain interview questions
LangGraph interview questions
LangSmith interview questions
RAG questions
tool calling questions
production system design questions

Completion Rule
After completing each section, provide:
Files created/updated.
Concepts covered.
How to run examples.
Suggested git commit message.
Ask before moving to next section.