"""
Concept: Defining tools with the @tool decorator.

A "tool" is a Python function the LLM can request to call.
The @tool decorator does three things:
  1. Reads the function name → becomes the tool name.
  2. Reads the docstring   → becomes the tool description the model uses
                             to decide WHEN to call this tool.
  3. Reads type hints      → becomes the JSON schema for the arguments
                             the model must supply.

The model never executes the function directly — it returns a structured
tool_call request. Your code decides when and how to actually run it.
(That execution loop is built in section 05.)

Run:
    python 04_tool_calling/01_define_tool.py
"""

from langchain_core.tools import tool


# ── Minimal tool ──────────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a given city."""
    # Stub — real implementation would call a weather API
    return f"The weather in {city} is sunny and 22°C."


# ── Tool with multiple typed arguments ────────────────────────────────────────

@tool
def create_reminder(title: str, due_date: str, priority: str = "medium") -> str:
    """
    Create a reminder.

    Args:
        title:    Short description of what to remember.
        due_date: Date in YYYY-MM-DD format.
        priority: One of 'low', 'medium', 'high'. Defaults to 'medium'.
    """
    return f"Reminder created: '{title}' due {due_date} [{priority}]"


# ── Inspect what the model sees ───────────────────────────────────────────────

import json

print("=== get_weather ===")
print("Name       :", get_weather.name)
print("Description:", get_weather.description)
print("Args schema:")
print(json.dumps(get_weather.args_schema.model_json_schema(), indent=2))

print("\n=== create_reminder ===")
print("Name       :", create_reminder.name)
print("Description:", create_reminder.description)
print("Args schema:")
print(json.dumps(create_reminder.args_schema.model_json_schema(), indent=2))

# ── Call the tool directly (no LLM involved) ──────────────────────────────────
print("\n=== Direct calls ===")
print(get_weather.invoke({"city": "London"}))
print(create_reminder.invoke({"title": "Submit PR", "due_date": "2026-06-25"}))
