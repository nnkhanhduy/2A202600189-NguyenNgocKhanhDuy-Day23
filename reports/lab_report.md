# Day 08 Lab Report

## 1. Team / student

- Name: nnkhanhduy
- Repo/commit: https://github.com/nnkhanhduy/2A202600189-NguyenNgocKhanhDuy-Day23
- Date: 2026-05-11

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
| S01_simple | simple | simple | ✓ | 0 | 0 |
| S02_tool | tool | tool | ✓ | 0 | 0 |
| S03_missing | missing_info | missing_info | ✓ | 0 | 0 |
| S04_risky | risky | risky | ✓ | 0 | 2 |
| S05_error | error | error | ✓ | 4 | 0 |
| S06_delete | risky | risky | ✓ | 0 | 2 |
| S07_dead_letter | error | error | ✓ | 2 | 0 |
| S08_conflict_risky_beats_tool | risky | risky | ✓ | 0 | 1 |
| S09_conflict_tool_beats_error | tool | tool | ✓ | 0 | 0 |
| S10_short_risky_beats_missing | risky | risky | ✓ | 0 | 1 |
| S11_error_word_variant | error | error | ✓ | 2 | 0 |
| S12_multi_risky_keywords | risky | risky | ✓ | 0 | 1 |
| S13_simple_no_keywords | simple | simple | ✓ | 0 | 0 |

**Summary:**
- Total scenarios: 13
- Success rate: 100.00%
- Average nodes visited: 10.31
- Total retries: 8
- Total interrupts: 7

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

- `S05_error`: transient failure attempt=1/3
- `S05_error`: transient failure attempt=2/3
- `S05_error`: transient failure attempt=1/3
- `S05_error`: transient failure attempt=2/3
- `S07_dead_letter`: transient failure attempt=1/1
- `S07_dead_letter`: transient failure attempt=1/1
- `S11_error_word_variant`: transient failure attempt=1/3
- `S11_error_word_variant`: transient failure attempt=2/3

## 6. Persistence / recovery evidence

SQLite checkpointer active. State survives process restart via `checkpoints.db`. Thread IDs are stable per scenario (`thread-<scenario_id>`), enabling crash-resume.

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

### Graph diagram (Mermaid)
Run `python -m langgraph_agent_lab.cli make-diagram` to export the graph as a Mermaid diagram
to `outputs/graph.md`.

### Time travel
Run `python -m langgraph_agent_lab.cli time-travel --thread-id <thread_id>` to replay
state history checkpoints for any completed scenario run.

## 8. Improvement plan

If given one more day:
1. **LLM-based routing**: Replace keyword heuristics in `classify_node` with a small LLM call
   (e.g., `claude-haiku-4-5`) for more robust intent detection — especially for ambiguous queries.
2. **Real tool integration**: Replace mock tool results with actual API calls (order management,
   CRM) with proper error handling and idempotency keys.
3. **Structured evaluation**: Replace `"ERROR:" in result` heuristic in `evaluate_node` with
   a structured response schema from the tool, so evaluation is deterministic.
4. **Streamlit HITL UI**: Build a minimal approval interface that receives the interrupt payload
   and lets a human approve/reject with a comment before resuming the graph.
