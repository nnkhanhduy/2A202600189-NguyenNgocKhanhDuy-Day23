"""Crash-resume demo using SQLite checkpointer.

Shows that state survives a simulated crash (KeyboardInterrupt mid-stream)
and can be resumed from the last saved checkpoint.

Run with:
    python crash_resume_demo.py
"""

from __future__ import annotations

import uuid
from src.langgraph_agent_lab.graph import build_graph
from src.langgraph_agent_lab.persistence import build_checkpointer
from src.langgraph_agent_lab.scenarios import Scenario
from src.langgraph_agent_lab.state import Route, initial_state

# Fresh thread_id each run so it doesn't conflict with completed threads in db
THREAD_ID = f"crash-demo-{uuid.uuid4().hex[:8]}"
DB_PATH = "checkpoints.db"


def print_separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def show_checkpoint(graph, config: dict) -> None:
    """Print current checkpoint state."""
    snap = graph.get_state(config)
    values = snap.values
    print(f"  route       : {values.get('route', '-')}")
    print(f"  attempt     : {values.get('attempt', 0)}")
    print(f"  final_answer: {str(values.get('final_answer', '-'))[:60]}")
    print(f"  next nodes  : {snap.next}")
    events = values.get("events", [])
    print(f"  nodes visited ({len(events)}): {[e['node'] for e in events]}")


# ---------------------------------------------------------------------------
# PHASE 1 — Run graph, crash after 3 nodes
# ---------------------------------------------------------------------------
print_separator("PHASE 1 — Run graph, simulate crash after 3 nodes")

checkpointer = build_checkpointer("sqlite", DB_PATH)
graph = build_graph(checkpointer=checkpointer)

scenario = Scenario(
    id=THREAD_ID,
    query="Timeout failure while processing request",
    expected_route=Route.ERROR,
    should_retry=True,
)
state = initial_state(scenario)
config = {"configurable": {"thread_id": THREAD_ID}}

print(f"\nStarting graph for thread_id={THREAD_ID!r} ...")
nodes_seen = 0
try:
    for chunk in graph.stream(state, config=config, stream_mode="values"):
        events = chunk.get("events", [])
        if events:
            node = events[-1]["node"]
            msg = events[-1]["message"]
            print(f"  [node={node}] {msg}")
            nodes_seen += 1
            if nodes_seen >= 3:
                raise KeyboardInterrupt("*** SIMULATED CRASH after 3 nodes ***")
except KeyboardInterrupt as e:
    print(f"\n  {e}")

print("\nCheckpoint state after crash:")
show_checkpoint(graph, config)

# ---------------------------------------------------------------------------
# PHASE 2 — New process: load from checkpoint, resume
# ---------------------------------------------------------------------------
print_separator("PHASE 2 — New process: load checkpoint and resume")

# Simulate a new process by rebuilding graph from scratch with same DB
checkpointer2 = build_checkpointer("sqlite", DB_PATH)
graph2 = build_graph(checkpointer=checkpointer2)

print(f"\nReloaded graph from {DB_PATH!r}")
print(f"Checkpoint state before resume:")
show_checkpoint(graph2, config)

print("\nResuming from checkpoint (same thread_id, no re-running completed nodes)...")
final_state = graph2.invoke(None, config=config)

print("\nFinal state after resume:")
print(f"  route       : {final_state.get('route')}")
print(f"  attempt     : {final_state.get('attempt', 0)}")
print(f"  final_answer: {final_state.get('final_answer', '-')}")
events = final_state.get("events", [])
print(f"  total nodes : {len(events)} — {[e['node'] for e in events]}")

# ---------------------------------------------------------------------------
# PHASE 3 — Show full state history (time travel)
# ---------------------------------------------------------------------------
print_separator("PHASE 3 — State history (time travel)")

history = list(graph2.get_state_history(config))
print(f"\n{len(history)} checkpoints saved for thread_id={THREAD_ID!r}:\n")
for snap in reversed(history):
    step = snap.metadata.get("step", "?")
    nodes = [e["node"] for e in snap.values.get("events", [])]
    route = snap.values.get("route", "-")
    print(f"  step={step:3}  next={snap.next}  route={route!r}  nodes_so_far={nodes}")

print("\nDone. Crash-resume demonstrated successfully.")
