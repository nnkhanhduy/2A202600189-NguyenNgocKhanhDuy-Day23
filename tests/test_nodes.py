"""Unit tests for individual node functions in nodes.py."""

from langgraph_agent_lab.nodes import (
    answer_node,
    ask_clarification_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    intake_node,
    retry_or_fallback_node,
    risky_action_node,
)
from langgraph_agent_lab.state import Route


# ---------------------------------------------------------------------------
# intake_node
# ---------------------------------------------------------------------------
class TestIntakeNode:
    def test_normalizes_query(self):
        result = intake_node({"query": "  How do I reset my password?  "})
        assert result["query"] == "How do I reset my password?"

    def test_emits_intake_event(self):
        result = intake_node({"query": "test query"})
        assert any(e["node"] == "intake" for e in result["events"])

    def test_empty_query(self):
        result = intake_node({"query": ""})
        assert result["query"] == ""
        assert result["events"]

    def test_adds_message(self):
        result = intake_node({"query": "hello world"})
        assert any("intake:" in m for m in result["messages"])


# ---------------------------------------------------------------------------
# ask_clarification_node
# ---------------------------------------------------------------------------
class TestAskClarificationNode:
    def test_sets_pending_question(self):
        result = ask_clarification_node({"query": "Fix it"})
        assert result["pending_question"]
        assert len(result["pending_question"]) > 10

    def test_sets_final_answer_to_question(self):
        result = ask_clarification_node({"query": "Update it now"})
        assert result["final_answer"] == result["pending_question"]

    def test_very_short_query(self):
        result = ask_clarification_node({"query": "Fix"})
        assert "detail" in result["pending_question"].lower() or "describe" in result["pending_question"].lower()

    def test_query_with_it_pronoun(self):
        result = ask_clarification_node({"query": "Can you fix it?"})
        assert "it" in result["pending_question"].lower() or "clarify" in result["pending_question"].lower()

    def test_emits_clarify_event(self):
        result = ask_clarification_node({"query": "Fix it"})
        assert any(e["node"] == "clarify" for e in result["events"])


# ---------------------------------------------------------------------------
# retry_or_fallback_node
# ---------------------------------------------------------------------------
class TestRetryOrFallbackNode:
    def test_increments_attempt(self):
        result = retry_or_fallback_node({"attempt": 0, "max_attempts": 3})
        assert result["attempt"] == 1

    def test_increments_from_nonzero(self):
        result = retry_or_fallback_node({"attempt": 2, "max_attempts": 3})
        assert result["attempt"] == 3

    def test_records_error(self):
        result = retry_or_fallback_node({"attempt": 1, "max_attempts": 3})
        assert result["errors"]
        assert "attempt=" in result["errors"][0]

    def test_emits_retry_event(self):
        result = retry_or_fallback_node({"attempt": 0, "max_attempts": 3})
        assert any(e["node"] == "retry" for e in result["events"])


# ---------------------------------------------------------------------------
# evaluate_node
# ---------------------------------------------------------------------------
class TestEvaluateNode:
    def test_success_on_clean_result(self):
        result = evaluate_node({"tool_results": ["Order status: SHIPPED"]})
        assert result["evaluation_result"] == "success"

    def test_needs_retry_on_error_result(self):
        result = evaluate_node({"tool_results": ["ERROR: transient failure attempt=1"]})
        assert result["evaluation_result"] == "needs_retry"

    def test_needs_retry_on_transient_failure(self):
        result = evaluate_node({"tool_results": ["transient failure attempt=1 scenario=S05"]})
        assert result["evaluation_result"] == "needs_retry"

    def test_empty_tool_results_defaults_success(self):
        result = evaluate_node({"tool_results": []})
        assert result["evaluation_result"] == "success"

    def test_emits_evaluate_event(self):
        result = evaluate_node({"tool_results": ["ok"]})
        assert any(e["node"] == "evaluate" for e in result["events"])


# ---------------------------------------------------------------------------
# dead_letter_node
# ---------------------------------------------------------------------------
class TestDeadLetterNode:
    def test_sets_final_answer_with_manual_review(self):
        result = dead_letter_node({"attempt": 3, "scenario_id": "S07", "errors": ["err1"]})
        assert "manual review" in result["final_answer"].lower()

    def test_includes_attempt_count(self):
        result = dead_letter_node({"attempt": 2, "scenario_id": "S07", "errors": []})
        assert "2" in result["final_answer"]

    def test_emits_dead_letter_event(self):
        result = dead_letter_node({"attempt": 1, "scenario_id": "x", "errors": []})
        assert any(e["node"] == "dead_letter" for e in result["events"])


# ---------------------------------------------------------------------------
# risky_action_node
# ---------------------------------------------------------------------------
class TestRiskyActionNode:
    def test_sets_proposed_action(self):
        result = risky_action_node({"query": "Refund this customer", "risk_level": "high"})
        assert result["proposed_action"]
        assert len(result["proposed_action"]) > 10

    def test_refund_action_type(self):
        result = risky_action_node({"query": "Refund this customer", "risk_level": "high"})
        assert "refund" in result["proposed_action"].lower()

    def test_delete_action_type(self):
        result = risky_action_node({"query": "Delete customer account", "risk_level": "high"})
        assert "delete" in result["proposed_action"].lower()

    def test_cancel_action_type(self):
        result = risky_action_node({"query": "Cancel my subscription", "risk_level": "high"})
        assert "cancel" in result["proposed_action"].lower()

    def test_emits_risky_action_event(self):
        result = risky_action_node({"query": "Refund customer", "risk_level": "high"})
        assert any(e["node"] == "risky_action" for e in result["events"])


# ---------------------------------------------------------------------------
# answer_node
# ---------------------------------------------------------------------------
class TestAnswerNode:
    def test_uses_tool_result(self):
        result = answer_node({
            "tool_results": ["Order status: SHIPPED"],
            "route": Route.TOOL.value,
            "approval": None,
        })
        assert "SHIPPED" in result["final_answer"]

    def test_approved_answer_includes_reviewer(self):
        result = answer_node({
            "tool_results": ["Refund processed"],
            "route": Route.RISKY.value,
            "approval": {"approved": True, "reviewer": "alice"},
        })
        assert "alice" in result["final_answer"]

    def test_simple_password_answer(self):
        result = answer_node({
            "tool_results": [],
            "route": Route.SIMPLE.value,
            "query": "How do I reset my password?",
        })
        assert result["final_answer"]
        assert "password" in result["final_answer"].lower()

    def test_simple_generic_answer(self):
        result = answer_node({
            "tool_results": [],
            "route": Route.SIMPLE.value,
            "query": "How long does shipping take?",
        })
        assert result["final_answer"]

    def test_emits_answer_event(self):
        result = answer_node({"tool_results": [], "route": Route.SIMPLE.value})
        assert any(e["node"] == "answer" for e in result["events"])


# ---------------------------------------------------------------------------
# finalize_node
# ---------------------------------------------------------------------------
class TestFinalizeNode:
    def test_emits_finalize_event(self):
        result = finalize_node({"route": Route.SIMPLE.value, "final_answer": "done"})
        assert any(e["node"] == "finalize" for e in result["events"])

    def test_has_answer_metadata(self):
        result = finalize_node({"route": Route.TOOL.value, "final_answer": "ok"})
        ev = next(e for e in result["events"] if e["node"] == "finalize")
        assert ev["metadata"]["has_answer"] is True

    def test_no_answer_metadata(self):
        result = finalize_node({"route": Route.ERROR.value, "final_answer": None, "pending_question": None})
        ev = next(e for e in result["events"] if e["node"] == "finalize")
        assert ev["metadata"]["has_answer"] is False
