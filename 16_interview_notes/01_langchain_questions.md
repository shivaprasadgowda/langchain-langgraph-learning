# LangChain Interview Questions

---

## Q1. What problem does LangChain solve that the raw OpenAI SDK does not?

**Answer:**
The OpenAI SDK gives you a thin HTTP wrapper — you manage prompt construction, message history, output parsing, retries, and composition yourself. LangChain provides:

- **Message types** (`HumanMessage`, `SystemMessage`, `AIMessage`) with consistent `.type` / `.content` attributes across all model providers.
- **LCEL (LangChain Expression Language)** — composable pipelines with `|` that support `.invoke()`, `.batch()`, `.stream()`, and `.astream_events()` on any chain automatically.
- **Output parsers** — `StrOutputParser`, `with_structured_output()`, `PydanticOutputParser` — so the LLM response is typed Python, not raw JSON strings.
- **Prompt templates** — `ChatPromptTemplate.from_messages()` with variable interpolation, re-usable across prompts.
- **Provider abstraction** — swap `ChatOpenAI` for `ChatAnthropic` or `ChatGoogleGenerativeAI` with zero chain changes.
- **Automatic LangSmith tracing** — zero-code observability via env vars.

---

## Q2. Explain LCEL (LangChain Expression Language) and what the `|` operator does.

**Answer:**
LCEL is a declarative composition system where `|` chains `Runnable` objects. Each `Runnable` must implement `.invoke(input)`. The `|` operator creates a `RunnableSequence` that passes the output of the left side as the input to the right side.

```python
chain = prompt | llm | StrOutputParser()
# Equivalent to:
# output = StrOutputParser().invoke(llm.invoke(prompt.invoke(input)))
```

Every LCEL chain automatically gets:
- `.invoke(input)` — synchronous single call
- `.batch([input1, input2])` — parallel calls with thread pool
- `.stream(input)` — yield output chunks progressively
- `.ainvoke()` / `.abatch()` / `.astream()` — async versions

This means a chain built with a single LLM call and a chain with 10 steps have identical interfaces — callers don't care what's inside.

---

## Q3. What is the difference between `langchain`, `langchain-core`, and `langchain-openai`?

**Answer:**

| Package | Contains | Depends on |
|---------|----------|-----------|
| `langchain-core` | Base classes: `Runnable`, `BaseMessage`, `BaseLLM`, `BaseOutputParser`, LCEL, callbacks | Nothing (pure interfaces) |
| `langchain-openai` | `ChatOpenAI`, `OpenAIEmbeddings` — OpenAI-specific implementations | `langchain-core`, `openai` SDK |
| `langchain` | High-level chains (`LLMChain`), agents, memory, document loaders, text splitters | `langchain-core`, `langchain-community` |
| `langchain-community` | Third-party integrations (FAISS, Chroma, many loaders) | `langchain-core` |

In practice: import from `langchain-core` for base types, `langchain-openai` for the model, and `langchain` for high-level utilities. This keeps dependencies minimal.

---

## Q4. What is `MessagesPlaceholder` and when do you need it?

**Answer:**
`MessagesPlaceholder` is a slot in a `ChatPromptTemplate` that accepts a **list of messages** rather than a string. You use it to inject chat history into a prompt template:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a DevOps assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()
response = chain.invoke({
    "chat_history": [
        HumanMessage(content="What is a pod?"),
        AIMessage(content="A pod is the smallest deployable unit in Kubernetes."),
    ],
    "question": "How many containers can a pod have?",
})
```

Without `MessagesPlaceholder`, you'd have to manually concatenate history into a string — losing the structured message format that modern LLMs process better.

---

## Q5. How does `with_structured_output()` work and what does it use under the hood?

**Answer:**
`llm.with_structured_output(MyPydanticModel)` instructs the LLM to return JSON that matches the schema, then parses the JSON into a typed Python object.

Under the hood it uses one of two mechanisms (chosen automatically based on model support):

1. **Tool calling** (preferred) — the schema is converted to a JSON Schema function definition and passed as a `tool`. The model returns a `tool_calls` field with structured JSON, not raw text. This is more reliable because the model knows it must fill valid fields.

2. **JSON mode** — `response_format={"type": "json_object"}` tells the model to return valid JSON, then the output is parsed with Pydantic. Less reliable than tool calling because the model can still hallucinate field names.

```python
class Intent(BaseModel):
    intent: Literal["jira", "aws", "general"]
    confidence: float

structured_llm = llm.with_structured_output(Intent)
result: Intent = structured_llm.invoke("Create a Jira ticket for the login bug")
# result.intent == "jira"
# result.confidence == 0.97
```

---

## Q6. What is the difference between `.invoke()`, `.batch()`, and `.stream()`?

**Answer:**

| Method | Input | Output | Use case |
|--------|-------|--------|---------|
| `.invoke(input)` | Single dict | Single output | One request, synchronous |
| `.batch([i1, i2])` | List of dicts | List of outputs | Multiple independent requests, runs in parallel (thread pool) |
| `.stream(input)` | Single dict | Iterator of chunks | Token-by-token output for UI streaming |
| `.ainvoke()` / `.astream()` | Same | Async coroutine / async gen | FastAPI async handlers |

`.batch()` is more efficient than calling `.invoke()` in a loop because LangChain uses a `ThreadPoolExecutor` internally (max_workers=5 by default). For truly parallel async batch work, use `.abatch()`.

---

## Q7. What does `RunnablePassthrough` do and when is it useful?

**Answer:**
`RunnablePassthrough` passes its input through unchanged. It's used inside `RunnableParallel` to forward the original input alongside processed values:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

rag_chain = (
    RunnableParallel({
        "context":  retriever | format_docs,   # retrieve and format docs
        "question": RunnablePassthrough(),     # original question unchanged
    })
    | prompt    # receives {"context": "...", "question": "..."}
    | llm
    | StrOutputParser()
)
```

Without `RunnablePassthrough`, the question would be replaced by the retriever's output and lost. It's the idiomatic way to branch a pipeline where one branch transforms the input and another keeps it as-is.

---

## Q8. How do you add retry logic to a LangChain chain for API errors?

**Answer:**
Use `.with_retry()`:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Retry on rate limits and transient errors, up to 3 attempts
resilient_llm = llm.with_retry(
    retry_if_exception_type=(Exception,),
    stop_after_attempt=3,
    wait_exponential_jitter=True,   # exponential backoff with jitter
)

chain = prompt | resilient_llm | StrOutputParser()
```

Or use `tenacity` directly for more control:
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
def call_llm(messages):
    return llm.invoke(messages)
```

In production, also set `timeout=30` on `ChatOpenAI` to fail fast rather than waiting indefinitely for a hung request.
