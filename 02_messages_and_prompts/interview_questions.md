# Interview Questions — 02 Messages & Prompts

---

## Q1. What are the three message types in LangChain and what role does each play?

**Answer:**

| Type | Role | Sent by |
|------|------|---------|
| `SystemMessage` | Sets the model's persona, tone, and constraints. Processed before any user input. | Developer |
| `HumanMessage` | A user turn — the input the model is responding to. | End user |
| `AIMessage` | A model turn — the response. Also added to history for multi-turn context. | Model |

All three share `.content` (the text) and `.type` (the role string). `AIMessage` also carries `.tool_calls`, `.usage_metadata`, and `.response_metadata`.

---

## Q2. Why does a chat model have no memory between calls?

**Answer:**
LLMs are stateless — every `.invoke()` is an independent HTTP request. The model only sees what is in the `messages` list you send. Memory is an application-layer concern: you must append each `HumanMessage` and `AIMessage` to a list and re-send the full list on every turn. LangGraph's checkpointer (section 08) automates this so you don't manage the list manually in production.

---

## Q3. What is `ChatPromptTemplate` and why use it over raw f-strings?

**Answer:**
`ChatPromptTemplate` is a reusable, structured prompt definition. You declare variables once (`{domain}`, `{question}`) and fill them at runtime via `.invoke()` or `.format_messages()`.

Advantages over f-strings:
- Works directly in LCEL chains with `|`
- `.input_variables` is introspectable — useful for validation
- Separates prompt structure from call-site code
- Easy to version, test, and swap without touching logic

---

## Q4. What problem does `MessagesPlaceholder` solve?

**Answer:**
A `ChatPromptTemplate` is static — all its messages are known at definition time. `MessagesPlaceholder` adds a dynamic slot where you inject a variable-length list of messages (typically the chat history) at runtime.

Without it, you'd have to rebuild the template on every turn as the history grows. With it, the template stays fixed and you just pass the updated history list under the placeholder's variable name.

This is the standard pattern used by LangGraph memory and agent loops.

---

## Q5. What is LCEL and what does the `|` operator do?

**Answer:**
LCEL (LangChain Expression Language) is a composition protocol. Every component (`ChatPromptTemplate`, `ChatOpenAI`, output parsers, retrievers, tools) implements the `Runnable` interface, which provides `.invoke()`, `.stream()`, and `.batch()`.

The `|` operator chains runnables so the output of the left side becomes the input of the right side:

```
prompt | model | parser
```

This is equivalent to:
```python
parser.invoke(model.invoke(prompt.invoke(input)))
```

but is lazy, composable, and automatically supports streaming and batching.

---

## Q6. What does `StrOutputParser` do and when would you use a different parser?

**Answer:**
`StrOutputParser` extracts `.content` from an `AIMessage` and returns a plain `str`. Use it when the caller just needs the text and shouldn't know about LangChain internals.

Alternative parsers:

| Parser | Use case |
|--------|---------|
| `StrOutputParser` | Plain text reply |
| `JsonOutputParser` | Model returns JSON — parse to `dict` |
| `PydanticOutputParser` | Validate JSON against a Pydantic schema |
| `CommaSeparatedListOutputParser` | Model returns a comma-separated list |

Section 03 covers `with_structured_output`, which is the modern preferred approach for structured replies.

---

## Q7. What is the difference between `.invoke()`, `.batch()`, and `.stream()` on a chain?

**Answer:**

| Method | Input | Output | Use case |
|--------|-------|--------|---------|
| `.invoke(dict)` | Single input dict | Single result | Default — one request |
| `.batch([dict, ...])` | List of input dicts | List of results | Run the chain on many inputs concurrently |
| `.stream(dict)` | Single input dict | Iterator of chunks | Stream tokens to the UI as they arrive |

All three are available on any `Runnable` in the chain, including the full composed chain, making it trivial to switch between modes without restructuring code.

---

## Q8. How does the order of messages in the list affect model behaviour?

**Answer:**
Order matters significantly:
1. `SystemMessage` must come first — it anchors the model's behaviour for the entire conversation.
2. `HumanMessage` / `AIMessage` pairs must alternate in chronological order — interleaving them incorrectly confuses the model about who said what.
3. The most recent `HumanMessage` at the end is what the model is currently responding to.

Sending history out of order or omitting earlier turns can cause the model to "forget" context, contradict itself, or produce inconsistent tone.
