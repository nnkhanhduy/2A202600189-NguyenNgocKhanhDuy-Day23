"""Full graph route tests covering all 5 routes + retry loop + dead letter."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None,
    reason="langgraph not installed",
)

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.scenarios import Scenario
from langgraph_agent_lab.state import Route, initial_state


def run(query: str, route: Route, **kwargs) -> dict:
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="test", query=query, expected_route=route, **kwargs)
    state = initial_state(scenario)
    return graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})


# ---------------------------------------------------------------------------
# All 5 routes reach finalize with an answer or question
# ---------------------------------------------------------------------------
class TestAllRoutes:
    def test_simple_route(self):
        result = run("How do I reset my password?", Route.SIMPLE)
        assert result["route"] == Route.SIMPLE.value
        assert result["final_answer"]

    def test_tool_route(self):
        result = run("Lookup order status for order 12345", Route.TOOL)
        assert result["route"] == Route.TOOL.value
        assert result["final_answer"]

    def test_missing_info_route(self):
        result = run("Can you fix it?", Route.MISSING_INFO)
        assert result["route"] == Route.MISSING_INFO.value
        assert result["pending_question"]

    def test_risky_route(self):
        result = run("Refund this customer", Route.RISKY)
        assert result["route"] == Route.RISKY.value
        assert result["final_answer"]
        assert result["approval"] is not None
        assert result["approval"]["approved"] is True

    def test_error_route_retries_then_succeeds(self):
        result = run("Timeout failure while processing", Route.ERROR, should_retry=True)
        assert result["route"] == Route.ERROR.value
        assert result["final_answer"]
        assert result["attempt"] >= 2  # retried at least twice


# ---------------------------------------------------------------------------
# Retry loop is bounded by max_attempts
# ---------------------------------------------------------------------------
class TestRetryLoop:
    def test_dead_letter_on_max_attempts_1(self):
        result = run(
            "System failure cannot recover after multiple attempts",
            Route.ERROR,
            max_attempts=1,
        )
        assert result["route"] == Route.ERROR.value
        assert result["final_answer"]
        assert "manual review" in result["final_answer"].lower()

    def test_retry_count_bounded(self):
        result = run("Timeout failure", Route.ERROR, max_attempts=2)
        assert result["attempt"] <= 2

    def test_error_events_recorded(self):
        result = run("Timeout failure", Route.ERROR, max_attempts=1)
        node_names = [e["node"] for e in result["events"]]
        assert "retry" in node_names
        assert "dead_letter" in node_names


# ---------------------------------------------------------------------------
# Risky path approval
# ---------------------------------------------------------------------------
class TestRiskyApproval:
    def test_approval_recorded(self):
        result = run("Delete customer account", Route.RISKY, requires_approval=True)
        assert result["approval"] is not None

    def test_proposed_action_set(self):
        result = run("Cancel subscription and remove data", Route.RISKY)
        assert result["proposed_action"] is not None
        assert len(result["proposed_action"]) > 0


# ---------------------------------------------------------------------------
# Node audit trail
# ---------------------------------------------------------------------------
class TestAuditTrail:
    def test_all_paths_include_finalize(self):
        for query, route in [
            ("Reset password", Route.SIMPLE),
            ("Lookup order 1", Route.TOOL),
            ("Can you fix it?", Route.MISSING_INFO),
            ("Refund customer", Route.RISKY),
        ]:
            result = run(query, route)
            nodes = [e["node"] for e in result["events"]]
            assert "finalize" in nodes, f"finalize missing for route={route}"

    def test_events_are_append_only(self):
        result = run("Timeout failure", Route.ERROR, max_attempts=2)
        nodes = [e["node"] for e in result["events"]]
        assert nodes[0] == "intake"
        assert nodes[-1] == "finalize"
