# 00 — Setup

> Status: **Complete**

Everything you need to get the repo running locally.

---

## Prerequisites

- Python 3.11 or higher (`python3 --version`)
- An [OpenAI API key](https://platform.openai.com/api-keys)
- (Optional) A [LangSmith API key](https://smith.langchain.com) for tracing

---

## Step 1 — Create a Virtual Environment

Run these commands from the **repo root** (not inside `00_setup/`):

```bash
python3 -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (Command Prompt)
venv\Scripts\activate.bat
```

You should see `(venv)` prepended to your shell prompt when it is active.

> Deactivate any time with: `deactivate`

---

## Step 2 — Install Packages

```bash
pip install -r requirements.txt
```

Key packages installed:

| Package | Purpose |
|---------|---------|
| `langchain` | Core abstractions (chains, messages, prompts) |
| `langchain-openai` | OpenAI chat model integration |
| `langchain-community` | Community loaders and tools |
| `langgraph` | Graph-based agent orchestration |
| `langsmith` | Tracing and evaluation |
| `faiss-cpu` | Local vector store for RAG examples |
| `tiktoken` | Token counting |
| `pydantic` | Structured output schemas |
| `python-dotenv` | Load `.env` into environment |
| `fastapi` / `uvicorn` | Web layer (used in section 15) |
| `pytest` | Test runner |

---

## Step 3 — Configure Environment Variables

Copy the example file:

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
# Required
OPENAI_API_KEY=sk-...

# Optional — only needed for section 14 (LangSmith)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=langchain-langgraph-learning
```

> `.env` is listed in `.gitignore` — it will never be committed.

---

## Step 4 — Verify the Setup

```bash
cd 00_setup
python 01_verify_setup.py
```

Expected output (all checks passing):

```
[OK] Python version: 3.x.x
[OK] langchain imported
[OK] langchain_openai imported
[OK] langgraph imported
[OK] langsmith imported
[OK] pydantic imported
[OK] dotenv imported
[OK] OPENAI_API_KEY is set
[OK] Live API call succeeded — model responded
Setup is complete. You are ready to start section 01.
```

If the live call fails, see Troubleshooting below.

---

## How to Run Any Example

All examples follow the same pattern:

```bash
# From repo root, with venv active
python 01_langchain_basics/01_first_llm_call.py
```

Or navigate into the section folder:

```bash
cd 01_langchain_basics
python 01_first_llm_call.py
```

Every script loads the `.env` file automatically via `python-dotenv` — no extra steps needed.

---

## Troubleshooting

### `ModuleNotFoundError`
The virtual environment is not active or packages were not installed.
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### `AuthenticationError` / `OPENAI_API_KEY` not set
Check that `.env` exists at the repo root and that the key starts with `sk-`.
```bash
cat .env   # confirm the key is there
```

### `python3 -m venv` not found
Install Python 3.11+ from [python.org](https://www.python.org/downloads/) or via your package manager:
```bash
brew install python@3.11   # macOS
```

### `pip install` fails on `faiss-cpu`
On Apple Silicon you may need:
```bash
pip install faiss-cpu --no-binary faiss-cpu
```

### VS Code does not see the venv
Open the Command Palette → `Python: Select Interpreter` → choose the `venv` entry.

---

## Files in This Section

| File | Purpose |
|------|---------|
| `README.md` | This file — full setup guide |
| `01_verify_setup.py` | Smoke-test imports and live API call |
| `interview_questions.md` | Setup-related interview Q&A |
