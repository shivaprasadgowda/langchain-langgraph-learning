# Interview Questions — 01 LangChain Basics

---

## Q1. What is LangChain and why would you use it instead of calling the OpenAI SDK directly?

**Answer:**
LangChain is a framework that wraps LLM providers behind a common interface (`BaseChatModel`) and provides building blocks — chains, prompts, tools, agents, memory — that compose cleanly via LCEL (LangChain Expression Language).

Reasons to prefer it over the raw SDK:

| Concern | Raw OpenAI SDK | LangChain |
|---------|---------------|-----------|
| Provider lock-in | Tied to OpenAI types | Swap model with one import |
| Prompt management | Manual string formatting | `ChatPromptTemplate`, `MessagesPlaceholder` |
| Chaining steps | Custom glue code | `prompt \| model \| parser` |
| Agents / tools | Build from scratch | `bind_tools`, `ToolNode`, LangGraph |
| Observability | Manual logging | LangSmith tracing built-in |

For a one-off script the raw SDK is fine. For anything with multiple steps, providers, or agents, LangChain pays off quickly.

---

## Q2. What does `ChatOpenAI` do under the hood?

**Answer:**
`ChatOpenAI` is a `BaseChatModel` subclass that:
1. Accepts a list of LangChain `BaseMessage` objects.
2. Converts them to the OpenAI `messages` format.
3. Calls the OpenAI chat completions endpoint.
4. Wraps the response in an `AIMessage` and populates `usage_metadata` and `response_metadata`.

This translation layer is what lets the rest of LangChain (chains, agents) remain provider-agnostic.

---

## Q3. What fields does `AIMessage` carry and which ones matter most?

**Answer:**

| Field | Type | When you use it |
|-------|------|----------------|
| `.content` | `str` | Always — the model's text reply |
| `.tool_calls` | `list` | Section 04 — when the model calls a tool |
| `.usage_metadata` | `dict` | Cost tracking, context limit checks |
| `.response_metadata` | `dict` | `finish_reason`, model name, raw provider data |
| `.id` | `str` | Correlating with provider logs |
| `.type` | `str` | Always `"ai"` — useful in type guards |

---

## Q4. What does `temperature` control and what value should you use for structured output?

**Answer:**
`temperature` scales the probability distribution over the model's next-token choices.

- `0.0` — always picks the highest-probability token (deterministic / greedy).
- `0.7` — default, balanced.
- `1.0+` — flattened distribution, more surprising/creative output.

For structured output (JSON extraction, classification, routing), use `temperature=0`. The task has one correct answer; randomness only introduces errors.

---

## Q5. What does `finish_reason: "length"` mean in the response metadata?

**Answer:**
The model stopped because it hit the `max_tokens` limit, not because it naturally finished. The output is likely truncated mid-sentence. In production this is a bug to fix — either raise `max_tokens` or constrain the prompt so the model can complete its answer within the limit.

`finish_reason: "stop"` means the model completed normally.

---

## Q6. How do you switch from OpenAI to Anthropic Claude with minimal code changes?

**Answer:**
```python
# Before
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# After
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
```

Everything downstream — `.invoke()`, chains, agents — stays the same because both implement `BaseChatModel`. This provider-agnosticism is one of LangChain's core value propositions.

---

## Q7. What is LCEL and how does it relate to what you learned in this section?

**Answer:**
LCEL (LangChain Expression Language) is a `|` pipe syntax where each component is a `Runnable` with `.invoke()`, `.stream()`, and `.batch()`. `ChatOpenAI` is a `Runnable`, so it can sit in a pipeline:

```python
chain = prompt | llm | output_parser
chain.invoke({"question": "What is LangChain?"})
```

Section 01 introduces `llm.invoke()` in isolation. Section 02 adds prompts and parsers to build the full chain.
