# 02 — Messages & Prompts

> Status: **Complete**

## What This Section Covers

- `SystemMessage`, `HumanMessage`, `AIMessage`
- Building a manual chat history
- `ChatPromptTemplate` and `MessagesPlaceholder`
- LCEL chain: `prompt | model | parser`

## Files

| File | Purpose |
|------|---------|
| `01_message_types.py` | The three message classes — roles, attributes, usage |
| `02_manual_chat_history.py` | Build and maintain history by hand across turns |
| `03_chat_prompt_template.py` | Reusable template with `{variables}`, used in LCEL |
| `04_messages_placeholder.py` | Inject a dynamic history list into a fixed template |
| `05_lcel_chain.py` | Full `prompt \| model \| parser` chain with `.invoke()`, `.batch()`, `.stream()` |
| `interview_questions.md` | 8 interview Q&As for this section |

## How to Run

Each file is independently runnable. From the repo root with `venv` active:

```bash
python 02_messages_and_prompts/01_message_types.py
python 02_messages_and_prompts/02_manual_chat_history.py
python 02_messages_and_prompts/03_chat_prompt_template.py
python 02_messages_and_prompts/04_messages_placeholder.py
python 02_messages_and_prompts/05_lcel_chain.py
```

## Key Concepts

**Message roles:** `SystemMessage` (persona/instructions) → `HumanMessage` (user turn) → `AIMessage` (model reply).

**Manual history:** append each `HumanMessage` + `AIMessage` to a list; re-send the full list every turn — the model has no built-in memory.

**`ChatPromptTemplate`:** declare `{variables}` once, fill at call time; chains directly with `|`.

**`MessagesPlaceholder`:** a slot in a template for a dynamic list of messages (history). The pattern used by every LangGraph agent.

**LCEL `|`:** `prompt | model | parser` — uniform `.invoke()` / `.batch()` / `.stream()` on the composed chain.
