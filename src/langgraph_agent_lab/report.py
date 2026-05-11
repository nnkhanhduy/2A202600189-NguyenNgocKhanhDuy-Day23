"""Report generation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a full lab report using the scenario metrics collected."""
    today = date.today().isoformat()

    # Build scenario table rows
    rows = []
    for m in metrics.scenario_metrics:
        status = "✓" if m.success else "✗"
        rows.append(
            f"| {m.scenario_id} | {m.expected_route} | {m.actual_route or '?'} "
            f"| {status} | {m.retry_count} | {m.interrupt_count} |"
        )
    scenario_table = "\n".join(rows)

    # Build error summary
    all_errors = []
    for m in metrics.scenario_metrics:
        for e in m.errors:
            all_errors.append(f"- `{m.scenario_id}`: {e}")
    error_section = "\n".join(all_errors) if all_errors else "- No errors recorded."

    resume_evidence = (
        "SQLite checkpointer active. State survives process restart via `checkpoints.db`. "
        "Thread IDs are stable per scenario (`thread-<scenario_id>`), enabling crash-resume."
        if metrics.resume_success
        else "MemorySaver checkpointer used. See bonus section for SQLite evidence."
    )

    return f"""# Day 08 Lab Report

## 1. Team / student

- Name: nnkhanhduy
- Repo/commit: phase2-track3-day8-langgraph-agent
- Date: {today}

## 2. Architecture

The graph is a LangGraph `StateGraph` with 11 nodes and conditional routing:

```
START → intake → classify → [conditional]
  simple       → answer → finalize → END
  tool         → tool → evaluate → answer → finalize → END
  tool (retry) → tool → evaluate → retry → tool → ... (bounded by max_attempts)
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → tool → evaluate → answer → finalize → END
  error        → retry → tool → evaluate → [retry loop or dead_letter]
  max retry    → dead_letter → finalize → END
```

Key design decisions:
- **Keyword priority**: risky > tool > missing_info > error > simple — prevents conflicts when queries match multiple categories.
- **Retry loop**: `evaluate_node` acts as the "done?" gate. If tool result starts with `ERROR:`, it sets `evaluation_result="needs_retry"` and `route_after_evaluate` loops back to `retry`.
- **Bounded retry**: `route_after_retry` checks `attempt >= max_attempts` and escalates to `dead_letter` on exhaustion.
- **HITL approval**: `approval_node` supports real `interrupt()` via `LANGGRAPH_INTERRUPT=true` env var; defaults to mock approval for CI.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| `query` | overwrite | normalized once in intake |
| `route` | overwrite | current classification only |
| `risk_level` | overwrite | set by classify, read by risky_action |
| `attempt` | overwrite | monotonically increasing retry counter |
| `max_attempts` | overwrite | scenario-level retry limit |
| `final_answer` | overwrite | last answer wins |
| `pending_question` | overwrite | last clarification question |
| `proposed_action` | overwrite | latest proposed risky action |
| `approval` | overwrite | latest approval decision |
| `evaluation_result` | overwrite | latest evaluate gate result |
| `messages` | append (`add`) | conversation audit trail |
| `tool_results` | append (`add`) | all tool call results for grounding |
| `errors` | append (`add`) | full error history |
| `events` | append (`add`) | append-only audit log for metrics |

## 4. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
{scenario_table}

**Summary:**
- Total scenarios: {metrics.total_scenarios}
- Success rate: {metrics.success_rate:.2%}
- Average nodes visited: {metrics.avg_nodes_visited:.2f}
- Total retries: {metrics.total_retries}
- Total interrupts: {metrics.total_interrupts}

## 5. Failure analysis

**1. Retry / tool failure (S05, S07)**

`S05_error` simulates a transient tool failure. The `tool_node` returns `ERROR:` on attempts < 2,
`evaluate_node` detects this and sets `evaluation_result="needs_retry"`, `route_after_evaluate`
sends back to `retry`, which increments `attempt` and routes to `tool` again via `route_after_retry`.
On attempt 2, the tool succeeds and the loop exits.

`S07_dead_letter` sets `max_attempts=1`, so after 1 failed attempt `route_after_retry` returns
`dead_letter` instead of `tool`. The `dead_letter_node` logs the failure and routes to `finalize`.

**2. Risky action without approval (S04, S06)**

`S04_risky` and `S06_delete` route to `risky_action` → `approval`. In production with
`LANGGRAPH_INTERRUPT=true`, the graph pauses and waits for a human decision. If rejected, the
graph routes to `clarify` → `finalize` instead of executing the action. In lab mode, mock approval
auto-approves so the tool runs.

{error_section}

## 6. Persistence / recovery evidence

{resume_evidence}

Each scenario run uses a stable `thread_id = "thread-<scenario_id>"`. The checkpointer saves
state after every node. To demonstrate crash-resume:

```bash
# Run with SQLite
python -m langgraph_agent_lab.cli time-travel --thread-id thread-S01_simple

# Or demonstrate state history
python -m langgraph_agent_lab.cli time-travel --thread-id thread-S02_tool
```

## 7. Extension work

### SQLite persistence
Switched checkpointer from `MemorySaver` to `SqliteSaver` in `configs/lab.yaml`.
Fixed the API bug: original skeleton used `SqliteSaver.from_conn_string()` (returns a context manager),
replaced with `SqliteSaver(conn=sqlite3.connect(..., check_same_thread=False))` with WAL mode enabled.

### Real HITL with interrupt/resume
`approval_node` uses `langgraph.types.interrupt()` when `LANGGRAPH_INTERRUPT=true`. The graph pauses
mid-execution, saves state to SQLite, and waits for a human decision. Resumed via `Command(resume=decision)`.

### Streamlit HITL UI (`streamlit_app.py`)
Two-tab Streamlit app:
- **Tab 1 (Agent HITL)**: Load any scenario, run the graph, and — for risky routes — the UI pauses
  and presents an approval/reject interface. After decision, the graph resumes from the saved checkpoint.
  Displays route match metrics and full execution log.
- **Tab 2 (Crash Resume Demo)**: Simulates a process crash mid-execution (configurable crash point),
  shows the SQLite checkpoint state, then resumes the graph in a fresh process from the last checkpoint.

Run with: `streamlit run streamlit_app.py`

### Crash-resume demo (`crash_resume_demo.py`)
Standalone script that demonstrates three phases:
1. Run the error-route graph, raise `KeyboardInterrupt` after N nodes (simulated crash).
2. Rebuild graph from scratch with same SQLite DB and same `thread_id` — resumes without re-running completed nodes.
3. Print full `get_state_history()` output (time travel) showing all checkpoint snapshots.

Run with: `python crash_resume_demo.py`

### Graph diagram (Mermaid)
Run `python -m langgraph_agent_lab.cli make-diagram` to export the graph as a Mermaid diagram
to `outputs/graph.md`.

### Time travel
Run `python -m langgraph_agent_lab.cli time-travel --thread-id <thread_id>` to replay
state history checkpoints for any completed scenario run.

### Extended test coverage (84 tests)
Added `tests/test_classify.py` (27 tests) covering all 5 routes, priority conflicts
(risky > tool > missing_info > error), word variant matching, and edge cases.
Added `tests/test_graph_full.py` (12 tests) covering all routes end-to-end, retry loop bounding,
dead-letter escalation, and audit trail completeness.
Added `tests/test_nodes.py` (34 tests) covering all node functions in isolation.

## 8. Improvement plan

If given one more day:
1. **LLM-based routing**: Replace keyword heuristics in `classify_node` with a small LLM call
   (e.g., `claude-haiku-4-5`) for more robust intent detection — especially for ambiguous queries.
2. **Real tool integration**: Replace mock tool results with actual API calls (order management,
   CRM) with proper error handling and idempotency keys.
3. **Structured evaluation**: Replace `"ERROR:" in result` heuristic in `evaluate_node` with
   a structured response schema from the tool, so evaluation is deterministic.
4. **Reject/edit outcomes in approval**: Currently rejected actions route to `clarify`. A richer
   flow would let the human edit the proposed action before re-approving.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
