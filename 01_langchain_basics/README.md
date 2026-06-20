# 01 — LangChain Basics

> Status: **Complete**

## What This Section Covers

- What is LangChain and why use it over the raw OpenAI SDK
- `ChatOpenAI` configuration
- Making a first LLM call
- Understanding the `AIMessage` response object

## Files

| File | Purpose |
|------|---------|
| `01_first_llm_call.py` | Minimal `ChatOpenAI` call — the absolute minimum to get a response |
| `02_ai_message_response.py` | Inspect every field of the `AIMessage` object |
| `03_chat_openai_config.py` | `temperature`, `max_tokens`, `timeout` — when and why to set each |
| `interview_questions.md` | 7 interview Q&As for this section |

## How to Run

Each file is independently runnable. From the repo root with `venv` active:

```bash
python 01_langchain_basics/01_first_llm_call.py
python 01_langchain_basics/02_ai_message_response.py
python 01_langchain_basics/03_chat_openai_config.py
```

Expected output for `01_first_llm_call.py`:
```
LangChain is a framework that simplifies building applications powered by large language models.
```

Expected output for `02_ai_message_response.py`:
```
content        : <model reply>
type           : ai
id             : chatcmpl-...
input_tokens   : <n>
output_tokens  : <n>
total_tokens   : <n>
finish_reason  : stop
model          : gpt-4o-mini-...
tool_calls     : []
```

## Key Concepts

**Why LangChain over raw OpenAI SDK?**
The SDK ties you to OpenAI-specific types. LangChain wraps every provider behind `BaseChatModel` — swap one import to change models, and chains/agents work unchanged.

**`ChatOpenAI` defaults:** `model="gpt-4o-mini"`, `temperature=0.7`

**`AIMessage` fields to know:** `.content`, `.tool_calls`, `.usage_metadata`, `.response_metadata`

**`temperature=0`** for classification/structured output. **`temperature>0`** for creative tasks.
