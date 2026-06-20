"""
Smoke-test for the repo environment.

Checks:
  1. Python version is 3.11+
  2. All required packages can be imported
  3. OPENAI_API_KEY is present in .env
  4. A live ChatOpenAI call succeeds

Run:
    python 00_setup/01_verify_setup.py
"""

import sys
import os

# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #

def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# 1. Python version
# --------------------------------------------------------------------------- #

major, minor = sys.version_info[:2]
if (major, minor) < (3, 11):
    fail(f"Python 3.11+ required, found {major}.{minor}")
ok(f"Python version: {major}.{minor}.{sys.version_info[2]}")

# --------------------------------------------------------------------------- #
# 2. Package imports
# --------------------------------------------------------------------------- #

packages = [
    ("langchain", "langchain"),
    ("langchain_openai", "langchain-openai"),
    ("langgraph", "langgraph"),
    ("langsmith", "langsmith"),
    ("pydantic", "pydantic"),
    ("dotenv", "python-dotenv"),
]

for module, pip_name in packages:
    try:
        __import__(module)
        ok(f"{pip_name} imported")
    except ImportError:
        fail(f"Cannot import '{module}'. Run: pip install {pip_name}")

# --------------------------------------------------------------------------- #
# 3. Load .env and check OPENAI_API_KEY
# --------------------------------------------------------------------------- #

# dotenv is confirmed importable above, so this is safe
from dotenv import load_dotenv

# load_dotenv searches upward from cwd, so it finds the root .env
# regardless of which directory you run this script from.
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key or api_key.startswith("sk-..."):
    fail(
        "OPENAI_API_KEY is missing or still set to the placeholder value.\n"
        "  1. Copy .env.example to .env\n"
        "  2. Replace sk-... with your real key"
    )
ok("OPENAI_API_KEY is set")

# --------------------------------------------------------------------------- #
# 4. Live API call
# --------------------------------------------------------------------------- #

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

try:
    # Use the cheapest/fastest model for the smoke test
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=10)
    response = llm.invoke([HumanMessage(content="Reply with the single word: ready")])
    ok(f"Live API call succeeded — model responded: '{response.content.strip()}'")
except Exception as e:
    fail(f"Live API call failed: {e}")

# --------------------------------------------------------------------------- #
# Done
# --------------------------------------------------------------------------- #

print()
print("Setup is complete. You are ready to start section 01.")
