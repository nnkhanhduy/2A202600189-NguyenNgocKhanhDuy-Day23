"""Streamlit HITL Approval UI for the LangGraph Support Agent.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import uuid

os.environ["LANGGRAPH_INTERRUPT"] = "true"

import streamlit as st
from langgraph.types import Command

from src.langgraph_agent_lab.graph import build_graph
from src.langgraph_agent_lab.persistence import build_checkpointer
from src.langgraph_agent_lab.scenarios import Scenario, load_scenarios
from src.langgraph_agent_lab.state import Route, initial_state

# ---------------------------------------------------------------------------
# Page config & cached resources
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Support Agent HITL", layout="wide")


@st.cache_resource
def get_graph():
    checkpointer = build_checkpointer("sqlite", "checkpoints.db")
    return build_graph(checkpointer=checkpointer)


@st.cache_data
def get_scenarios():
    return load_scenarios("data/sample/scenarios.jsonl")


graph = get_graph()
all_scenarios = get_scenarios()

ROUTE_BADGE = {
    "simple": "green",
    "tool": "blue",
    "missing_info": "orange",
    "risky": "red",
    "error": "violet",
    "error (dead_letter)": "gray",
}

# ---------------------------------------------------------------------------
# Sidebar — minimal legend only
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Route priority**")
    for route, color in ROUTE_BADGE.items():
        st.markdown(f":{color}[{route}]")
    st.divider()
    st.caption(f"{len(all_scenarios)} scenarios · SQLite checkpoint")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults: dict = {
    "phase": "input",
    "thread_id": None,
    "interrupt_payload": None,
    "final_state": None,
    "run_log": [],
    "expected_route": "simple",
    "requires_approval": False,
    # crash-resume demo state
    "cr_phase": "idle",   # idle | crashed | done
    "cr_thread_id": None,
    "cr_log_phase1": [],
    "cr_log_phase2": [],
    "cr_history": [],
    "cr_checkpoint_snap": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset() -> None:
    for k, v in defaults.items():
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------
def run_graph(query: str, expected_route: str, requires_approval: bool) -> None:
    thread_id = f"hitl-{uuid.uuid4().hex[:8]}"
    st.session_state.update(
        thread_id=thread_id, run_log=[],
        expected_route=expected_route, requires_approval=requires_approval,
    )
    scenario = Scenario(id=thread_id, query=query, expected_route=Route(expected_route))
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": thread_id}}

    for chunk in graph.stream(state, config=config, stream_mode="values"):
        for ev in chunk.get("events", []):
            entry = (ev["node"], ev["message"])
            if not st.session_state.run_log or st.session_state.run_log[-1] != entry:
                st.session_state.run_log.append(entry)

    current = graph.get_state(config)
    if current.next:
        if current.tasks and current.tasks[0].interrupts:
            st.session_state.interrupt_payload = current.tasks[0].interrupts[0].value
        st.session_state.phase = "waiting_approval"
    else:
        st.session_state.final_state = current.values
        st.session_state.phase = "done"


def resume_graph(approved: bool, comment: str) -> None:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    decision = {"approved": approved, "reviewer": "human-operator", "comment": comment}

    for chunk in graph.stream(Command(resume=decision), config=config, stream_mode="values"):
        for ev in chunk.get("events", []):
            entry = (ev["node"], ev["message"])
            if not st.session_state.run_log or st.session_state.run_log[-1] != entry:
                st.session_state.run_log.append(entry)

    st.session_state.final_state = graph.get_state(config).values
    st.session_state.phase = "done"


# ---------------------------------------------------------------------------
# Crash-resume helpers
# ---------------------------------------------------------------------------
def cr_run_crash(crash_after: int) -> None:
    thread_id = f"crash-{uuid.uuid4().hex[:8]}"
    st.session_state.cr_thread_id = thread_id
    st.session_state.cr_log_phase1 = []

    scenario = Scenario(
        id=thread_id,
        query="Timeout failure while processing request",
        expected_route=Route.ERROR,
    )
    config = {"configurable": {"thread_id": thread_id}}
    nodes_seen = 0

    try:
        for chunk in graph.stream(initial_state(scenario), config=config, stream_mode="values"):
            for ev in chunk.get("events", []):
                entry = (ev["node"], ev["message"])
                if not st.session_state.cr_log_phase1 or st.session_state.cr_log_phase1[-1] != entry:
                    st.session_state.cr_log_phase1.append(entry)
                    nodes_seen += 1
                    if nodes_seen >= crash_after:
                        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass

    snap = graph.get_state(config)
    st.session_state.cr_checkpoint_snap = {
        "route": snap.values.get("route", "-"),
        "attempt": snap.values.get("attempt", 0),
        "next": snap.next,
        "nodes": [e["node"] for e in snap.values.get("events", [])],
    }
    st.session_state.cr_phase = "crashed"


def cr_run_resume() -> None:
    thread_id = st.session_state.cr_thread_id
    config = {"configurable": {"thread_id": thread_id}}
    st.session_state.cr_log_phase2 = []

    # Build fresh graph (simulates new process) using same db
    fresh_graph = build_graph(checkpointer=build_checkpointer("sqlite", "checkpoints.db"))
    final = fresh_graph.invoke(None, config=config)

    for ev in final.get("events", []):
        entry = (ev["node"], ev["message"])
        if entry not in st.session_state.cr_log_phase1:
            if not st.session_state.cr_log_phase2 or st.session_state.cr_log_phase2[-1] != entry:
                st.session_state.cr_log_phase2.append(entry)

    history = list(fresh_graph.get_state_history(config))
    st.session_state.cr_history = [
        {
            "step": s.metadata.get("step", i),
            "next": s.next,
            "route": s.values.get("route", "-"),
            "nodes": [e["node"] for e in s.values.get("events", [])],
        }
        for i, s in enumerate(reversed(history))
    ]
    st.session_state.cr_phase = "done"


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.title("LangGraph Support Agent")
tab_agent, tab_crash = st.tabs(["Agent (HITL)", "Crash Resume Demo"])

# ===========================================================================
# TAB 1 — HITL Agent
# ===========================================================================
with tab_agent:

    # ---- INPUT ----
    if st.session_state.phase == "input":
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Submit a request")
            options = ["(type your own)"] + [
                f"[{s.expected_route.value}]  {s.id} — {s.query[:55]}"
                for s in all_scenarios
            ]
            selected = st.selectbox("Load a scenario", options)

            if selected != "(type your own)":
                idx = options.index(selected) - 1
                chosen = all_scenarios[idx]
                prefill_query = chosen.query
                prefill_route = chosen.expected_route.value
                prefill_approval = chosen.requires_approval
            else:
                prefill_query, prefill_route, prefill_approval = "", "simple", False

            query = st.text_area("Query", value=prefill_query, height=100,
                                 placeholder="Describe your support request...")

            if st.button("Run Agent", type="primary", disabled=not query.strip()):
                with st.spinner("Running..."):
                    run_graph(query.strip(), prefill_route, prefill_approval)
                st.rerun()

        with col_right:
            st.subheader("Routing rules")
            st.markdown("""
| Route | Keywords |
|---|---|
| `risky` | refund, delete, cancel, remove... |
| `tool` | status, order, find, check... |
| `missing_info` | < 5 words + "it" |
| `error` | timeout, crash, fail... |
| `simple` | everything else |

**Priority:** risky > tool > missing_info > error > simple
            """)
            st.info("Risky routes pause for human approval.")

    # ---- WAITING APPROVAL ----
    elif st.session_state.phase == "waiting_approval":
        payload = st.session_state.interrupt_payload or {}
        st.subheader("[PAUSED] Human Approval Required")
        st.warning("The agent paused and is waiting for your decision.")

        col1, col2 = st.columns([3, 2])
        with col1:
            with st.container(border=True):
                st.markdown("**Proposed Action**")
                st.write(payload.get("proposed_action", "No details."))
                risk = payload.get("risk_level", "unknown")
                st.markdown(f"**Risk Level:** `{risk.upper()}`")

            comment = st.text_input("Comment", placeholder="Optional note for audit trail...")
            b1, b2, b3 = st.columns([2, 2, 1])
            with b1:
                if st.button("Approve", type="primary", use_container_width=True):
                    with st.spinner("Resuming..."):
                        resume_graph(True, comment or "approved")
                    st.rerun()
            with b2:
                if st.button("Reject", type="secondary", use_container_width=True):
                    with st.spinner("Resuming..."):
                        resume_graph(False, comment or "rejected")
                    st.rerun()
            with b3:
                if st.button("Cancel"):
                    reset(); st.rerun()

        with col2:
            if st.session_state.run_log:
                st.markdown("**Log so far**")
                for node, msg in st.session_state.run_log:
                    st.text(f"[{node}] {msg}")

    # ---- DONE ----
    elif st.session_state.phase == "done":
        final = st.session_state.final_state or {}
        actual = final.get("route", "unknown")
        expected = st.session_state.expected_route
        answer = final.get("final_answer") or final.get("pending_question") or "—"
        approval = final.get("approval")
        match = actual == expected

        st.subheader("Completed")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Actual Route", actual)
        m2.metric("Expected Route", expected)
        m3.metric("Route Match", "PASS" if match else "FAIL")
        m4.metric("Retries", final.get("attempt", 0))

        if not match:
            st.error(f"Mismatch: expected `{expected}` got `{actual}`")

        col1, col2 = st.columns([3, 2])
        with col1:
            with st.container(border=True):
                st.markdown("**Agent Response**")
                st.write(answer)
            if approval:
                ok = approval.get("approved", False)
                msg = f"{'Approved' if ok else 'Rejected'} by `{approval.get('reviewer')}` — {approval.get('comment')}"
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
            if final.get("errors"):
                with st.expander(f"Errors ({len(final['errors'])})"):
                    for e in final["errors"]: st.code(e)

        with col2:
            if st.session_state.run_log:
                with st.expander("Execution log", expanded=True):
                    for i, (node, msg) in enumerate(st.session_state.run_log, 1):
                        st.text(f"{i:2d}. [{node}] {msg}")

        st.divider()
        if st.button("New Request", type="primary"):
            reset(); st.rerun()

# ===========================================================================
# TAB 2 — Crash Resume Demo
# ===========================================================================
with tab_crash:
    st.subheader("Crash Resume Demo")
    st.markdown("Simulates a process crash mid-execution, then resumes from the last SQLite checkpoint.")

    # ---- IDLE ----
    if st.session_state.cr_phase == "idle":
        crash_after = st.slider("Crash after N nodes", min_value=1, max_value=5, value=3)
        st.caption("Graph has ~10 nodes for error route. Crashing after 3 stops it mid-retry-loop.")
        if st.button("Run & Crash", type="primary"):
            with st.spinner("Running until crash..."):
                cr_run_crash(crash_after)
            st.rerun()

    # ---- CRASHED ----
    elif st.session_state.cr_phase == "crashed":
        snap = st.session_state.cr_checkpoint_snap or {}

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Phase 1 — Crashed**")
            st.error(f"Process crashed after {len(st.session_state.cr_log_phase1)} nodes")
            for node, msg in st.session_state.cr_log_phase1:
                st.text(f"[{node}] {msg}")

        with col2:
            st.markdown("**Checkpoint saved in SQLite**")
            with st.container(border=True):
                st.markdown(f"route: `{snap.get('route')}`")
                st.markdown(f"attempt: `{snap.get('attempt')}`")
                st.markdown(f"next node: `{snap.get('next')}`")
                st.markdown(f"nodes done: `{snap.get('nodes')}`")

        st.divider()
        if st.button("Resume from checkpoint (new process)", type="primary"):
            with st.spinner("Resuming..."):
                cr_run_resume()
            st.rerun()

        if st.button("Reset"):
            st.session_state.cr_phase = "idle"
            st.rerun()

    # ---- DONE ----
    elif st.session_state.cr_phase == "done":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Phase 1 — Before crash**")
            for node, msg in st.session_state.cr_log_phase1:
                st.text(f"[{node}] {msg}")

            st.divider()
            st.markdown("**Phase 2 — Resumed (skipped completed nodes)**")
            if st.session_state.cr_log_phase2:
                for node, msg in st.session_state.cr_log_phase2:
                    st.text(f"[{node}] {msg}")
            else:
                st.caption("(no new events — all resumed from checkpoint)")

        with col2:
            st.markdown("**Full checkpoint history (time travel)**")
            history = st.session_state.cr_history
            for snap in history:
                st.text(
                    f"step={snap['step']:3d}  next={snap['next']}  "
                    f"route={snap['route']!r}  nodes={len(snap['nodes'])}"
                )

        st.success(f"Crash-resume complete. thread_id=`{st.session_state.cr_thread_id}`")
        if st.button("Run again", type="primary"):
            st.session_state.cr_phase = "idle"
            st.rerun()
