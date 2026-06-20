"""
Concept: PostgreSQL checkpointer for production persistence.

InMemorySaver (section 02) is lost when the process restarts.
In production you need a durable store so conversations survive:
  - Server restarts and deployments
  - Horizontal scaling (any replica can continue any thread)
  - Crash recovery (resume mid-graph from last completed node)

LangGraph provides AsyncPostgresSaver (and AsyncSqliteSaver for SQLite)
from the langgraph-checkpoint-postgres package.

This file is a code sketch with detailed comments — it does NOT connect
to a real database. Run it to see the pattern explained.

Install (not needed to run this file):
    pip install langgraph-checkpoint-postgres
    pip install psycopg[binary,pool]

Run:
    python 08_persistence_checkpointer/04_postgres_concept.py
"""


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ═══════════════════════════════════════════════════════════════════════════════
section("1. The production swap — only the checkpointer changes")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  Development (MemorySaver):
  ─────────────────────────
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)


  Production (AsyncPostgresSaver):
  ────────────────────────────────
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    import psycopg

    # Connection string from environment — never hardcode credentials
    DB_URL = os.environ["DATABASE_URL"]
    # e.g. "postgresql://user:pass@host:5432/dbname"

    async def create_app():
        async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()   # creates the checkpoints table
            app = graph.compile(checkpointer=checkpointer)
            return app

  Everything else — State, nodes, edges — stays identical.
  The checkpointer is the only change between dev and prod.
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("2. What LangGraph stores in the checkpoints table")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  After every node execution, LangGraph writes a checkpoint row:

    thread_id       — identifies the conversation (e.g. user session ID)
    checkpoint_id   — monotonically increasing within a thread
    parent_id       — previous checkpoint (enables branching / time-travel)
    state           — full serialised state dict (JSON)
    metadata        — node name, step count, timestamps

  On the next invoke() with the same thread_id, LangGraph:
    1. Queries for the latest checkpoint_id for that thread
    2. Deserialises the state
    3. Injects it as the starting state for the graph
    4. Runs only the nodes that haven't been completed yet

  This means a graph can survive a mid-execution crash and resume
  from the last completed node rather than restarting from scratch.
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("3. Connection pooling with psycopg_pool")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  A single async connection works in development but not under load.
  Use a connection pool in production:

    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    pool = AsyncConnectionPool(conninfo=DB_URL, max_size=20)
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

  The pool is created once at application startup and reused across
  all requests, avoiding the overhead of opening a new connection
  per invocation.
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("4. FastAPI integration pattern")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  from contextlib import asynccontextmanager
  from fastapi import FastAPI
  from psycopg_pool import AsyncConnectionPool
  from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

  pool = None
  app_graph = None

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      global pool, app_graph
      pool = AsyncConnectionPool(conninfo=os.environ["DATABASE_URL"])
      checkpointer = AsyncPostgresSaver(pool)
      await checkpointer.setup()
      app_graph = graph.compile(checkpointer=checkpointer)
      yield                          # application runs here
      await pool.close()             # cleanup on shutdown

  api = FastAPI(lifespan=lifespan)

  @api.post("/chat")
  async def chat(message: str, thread_id: str):
      config = {"configurable": {"thread_id": thread_id}}
      result = await app_graph.ainvoke(
          {"messages": [HumanMessage(content=message)]},
          config=config,
      )
      return {"reply": result["messages"][-1].content}
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("5. SQLite alternative for single-server deployments")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  If PostgreSQL is not available, SQLite provides durable persistence
  for single-server deployments (no horizontal scaling):

    pip install langgraph-checkpoint-sqlite

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        await checkpointer.setup()
        app = graph.compile(checkpointer=checkpointer)

  SQLite is fine for:
    - Local development with persistence
    - Single-node internal tools
    - Prototypes and demos

  Use PostgreSQL for:
    - Multi-replica deployments
    - High write throughput
    - Enterprise / production workloads
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("6. Time-travel and state inspection")
# ═══════════════════════════════════════════════════════════════════════════════
print("""
  Because every checkpoint is stored, you can:

  # Get current state
  snapshot = app.get_state(config)

  # List all checkpoints for a thread (most recent first)
  for checkpoint in app.get_state_history(config):
      print(checkpoint.config["configurable"]["checkpoint_id"])
      print(checkpoint.values["messages"])

  # Resume from an earlier checkpoint (time-travel)
  old_config = {"configurable": {"thread_id": "abc", "checkpoint_id": "old-id"}}
  app.invoke({"messages": [HumanMessage("start over from here")]}, config=old_config)

  This is invaluable for debugging, auditing, and A/B testing
  different agent behaviours on the same conversation history.
""")
