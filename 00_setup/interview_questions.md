# Interview Questions — 00 Setup

These questions test understanding of the environment and tooling decisions behind a LangChain / LangGraph project.

---

## Q1. Why use a virtual environment instead of installing packages globally?

**Answer:**
A virtual environment isolates project dependencies so different projects can use different (potentially conflicting) package versions. Without it, installing `langchain==0.2` for one project could break another that needs `langchain==0.3`. It also keeps the global Python installation clean and makes the project reproducible — anyone can recreate the exact environment with `pip install -r requirements.txt`.

---

## Q2. What is `python-dotenv` and why is it used here?

**Answer:**
`python-dotenv` reads a `.env` file and loads its key-value pairs into `os.environ` at runtime. This keeps secrets (API keys) out of source code while still making them available to the application. The `.env` file is listed in `.gitignore` so it is never committed. A committed `.env.example` shows collaborators which variables are required without exposing real values.

---

## Q3. Why keep `OPENAI_API_KEY` in `.env` instead of hardcoding it?

**Answer:**
Hardcoded secrets end up in version history and are exposed to anyone with repo access — including public forks. Environment variables separate configuration from code. If a key is compromised, you rotate it in one place without touching code. This follows the [12-factor app](https://12factor.net/config) principle: store config in the environment.

---

## Q4. What is the difference between `langchain`, `langchain-core`, and `langchain-openai`?

**Answer:**

| Package | Role |
|---------|------|
| `langchain-core` | Base abstractions: `BaseMessage`, `BaseLanguageModel`, LCEL `Runnable`, etc. Minimal dependencies. |
| `langchain` | Higher-level chains, agents, and utilities built on top of `langchain-core`. |
| `langchain-openai` | OpenAI-specific integration (`ChatOpenAI`, `OpenAIEmbeddings`). Depends on `openai` SDK. |

You can use `langchain-openai` + `langchain-core` alone for many tasks. `langchain` adds convenience layers on top.

---

## Q5. Why pin minimum versions (e.g., `langchain>=0.3.0`) instead of exact versions?

**Answer:**
Minimum version pins (`>=`) allow `pip` to install compatible upgrades (security patches, bug fixes) without breaking the API contract. Exact pins (`==`) guarantee full reproducibility but require manual maintenance. For a learning repo, minimum pins strike the right balance. In production you would typically generate a locked `requirements.lock` with exact versions after testing.

---

## Q6. What does `load_dotenv()` do if `.env` does not exist?

**Answer:**
It silently does nothing — it returns `False` but raises no exception. Variables already present in the environment (e.g., set by CI/CD or shell export) are unaffected. This makes `load_dotenv()` safe to call unconditionally; production environments can inject secrets through the real environment without needing a `.env` file.

---

## Q7. What is LangSmith and do you need it to run these examples?

**Answer:**
LangSmith is Anthropic's — actually LangChain's — observability platform. It records every LLM call, prompt, and tool invocation in a searchable trace. It is entirely optional for sections 01–13. Setting `LANGCHAIN_TRACING_V2=true` in `.env` activates it; omitting that variable means no tracing happens. Section 14 covers it in depth.

---

## Q8. Why is `faiss-cpu` listed in requirements even though it is only used in section 09?

**Answer:**
All dependencies are declared upfront so the environment can be set up once and every section runs without additional installs. This mirrors how production projects work — a single `requirements.txt` (or `pyproject.toml`) captures all dependencies at the start. It also avoids the confusion of per-section install steps.
