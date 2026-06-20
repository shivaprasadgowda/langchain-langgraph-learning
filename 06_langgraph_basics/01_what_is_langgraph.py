"""
Concept: What is LangGraph and why does it exist?

This file is a pure explanation — no API calls, no LLM.
Run it to see a structured walkthrough printed to the terminal.

The problem with the manual loop (section 05):
  - A while-loop is linear. Real agent behaviour branches, backtracks,
    pauses for human input, and runs steps in parallel.
  - There is no built-in persistence — a crash wipes all history.
  - Streaming, observability, and testing all require custom plumbing.

LangGraph models agent logic as a directed graph:
  - Nodes  = units of work (call LLM, run tool, classify, etc.)
  - Edges  = flow control (always go to X, or conditionally go to Y or Z)
  - State  = a dict that flows through every node and accumulates results

Key properties:
  1. Persistence  — a Checkpointer snapshots state after every node,
                    enabling resume after crash and multi-turn memory.
  2. Branching    — conditional edges let the graph route to different
                    nodes based on the current state.
  3. Cycles       — graphs can loop (model → tools → model) without a
                    hand-written while-loop.
  4. Streaming    — every node's output can be streamed to a UI.
  5. Human-in-the-loop — interrupt() pauses the graph at any node.

Run:
    python 06_langgraph_basics/01_what_is_langgraph.py
"""


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


section("LangGraph vs Manual Loop")
print("""
  Manual loop (section 05):

    messages = [...]
    while True:
        response = llm.invoke(messages)
        if not response.tool_calls: break
        ... dispatch tools ...

  Problems:
    - Linear — cannot branch to different handlers
    - No persistence — restart = lost context
    - Hard to pause for human approval
    - Difficult to stream to a UI
    - Monolithic — hard to test individual steps

  LangGraph:

    graph = StateGraph(State)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", route)
    graph.add_edge("tools", "llm")
    app = graph.compile(checkpointer=...)

    app.invoke({"messages": [HumanMessage(...)]})
""")


section("The Four Primitives")
print("""
  1. STATE
     A TypedDict (or Pydantic model) that holds everything the graph knows.
     Every node receives the full state and returns a partial update.
     LangGraph merges the update into the state automatically.

     Example:
       class State(TypedDict):
           messages: Annotated[list, add_messages]
           intent:   str

  2. NODE
     A plain Python function: (state) -> partial_state_update.
     Can call an LLM, run a tool, classify text, write to a DB — anything.

     Example:
       def call_llm(state: State) -> dict:
           response = llm.invoke(state["messages"])
           return {"messages": [response]}

  3. EDGE
     Defines which node runs next.
     - Normal edge: always go to node X.
     - Conditional edge: call a function on state, return node name.

     Example:
       graph.add_edge("tools", "llm")                    # normal
       graph.add_conditional_edges("llm", route_fn)      # conditional

  4. STATE REDUCER
     Decides how a node's partial update is merged into the existing state.
     The built-in add_messages reducer *appends* new messages rather than
     replacing the list — essential for chat history.

     Without a reducer: new value replaces old value.
     With add_messages:  new messages are appended to the existing list.
""")


section("Graph Lifecycle")
print("""
  1. Define State  →  TypedDict with field types and reducers
  2. Create graph  →  StateGraph(State)
  3. Add nodes     →  graph.add_node("name", fn)
  4. Add edges     →  graph.add_edge / add_conditional_edges
  5. Compile       →  app = graph.compile()   (optionally with checkpointer)
  6. Invoke        →  app.invoke(initial_state, config={"configurable": {"thread_id": "1"}})
  7. Stream        →  for event in app.stream(...): ...
""")


section("START and END")
print("""
  START  — a special constant representing the graph's entry point.
           graph.add_edge(START, "first_node") means "begin here".

  END    — a special constant representing a terminal state.
           Returning END from a conditional edge stops execution.
           A graph can have multiple paths to END.

  Import:
    from langgraph.graph import StateGraph, START, END
""")


section("Next Steps")
print("""
  02_first_graph.py  — build and run a minimal one-node graph
  03_state_node_edge.py — each primitive isolated and explained
""")
