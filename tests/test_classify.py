"""Tests for classify_node keyword routing and priority logic."""

from langgraph_agent_lab.nodes import classify_node
from langgraph_agent_lab.state import Route


def classify(query: str) -> str:
    return classify_node({"query": query})["route"]


# ---------------------------------------------------------------------------
# Basic routes
# ---------------------------------------------------------------------------
class TestBasicRoutes:
    def test_simple_no_keywords(self):
        assert classify("How long does standard shipping take?") == Route.SIMPLE.value

    def test_simple_password(self):
        assert classify("How do I reset my password?") == Route.SIMPLE.value

    def test_tool_order_status(self):
        assert classify("Please lookup order status for order 12345") == Route.TOOL.value

    def test_tool_find(self):
        assert classify("Find my recent purchase") == Route.TOOL.value

    def test_tool_check(self):
        assert classify("Check my delivery status") == Route.TOOL.value

    def test_missing_info_fix_it(self):
        assert classify("Can you fix it?") == Route.MISSING_INFO.value

    def test_missing_info_short_with_it(self):
        assert classify("Update it now") == Route.MISSING_INFO.value

    def test_risky_refund(self):
        assert classify("Refund this customer") == Route.RISKY.value

    def test_risky_delete(self):
        assert classify("Delete customer account") == Route.RISKY.value

    def test_risky_cancel(self):
        assert classify("Cancel my subscription") == Route.RISKY.value

    def test_error_timeout(self):
        assert classify("Timeout failure while processing request") == Route.ERROR.value

    def test_error_crash(self):
        assert classify("System crash unavailable") == Route.ERROR.value


# ---------------------------------------------------------------------------
# Priority conflicts (risky > tool > missing_info > error > simple)
# ---------------------------------------------------------------------------
class TestPriorityConflicts:
    def test_risky_beats_tool(self):
        """'check' is tool keyword, 'refund' is risky — risky must win."""
        assert classify("Check if the refund was processed for order 99999") == Route.RISKY.value

    def test_tool_beats_error(self):
        """'error' keyword present but 'find' and 'order' are tool — tool must win."""
        assert classify("Find the error logs for order 5678") == Route.TOOL.value

    def test_risky_beats_missing_info(self):
        """Short query with 'it' would be missing_info, but 'delete' makes it risky."""
        assert classify("Delete it all") == Route.RISKY.value

    def test_risky_beats_error(self):
        """Both 'cancel' (risky) and 'failed' (error) present — risky must win."""
        assert classify("Cancel the failed transaction") == Route.RISKY.value

    def test_multi_risky_keywords(self):
        assert classify("Cancel my subscription and remove all my data") == Route.RISKY.value


# ---------------------------------------------------------------------------
# Word variant / prefix matching
# ---------------------------------------------------------------------------
class TestWordVariants:
    def test_crashing_matches_error(self):
        assert classify("The server keeps crashing") == Route.ERROR.value

    def test_failing_matches_error(self):
        assert classify("Requests are failing constantly") == Route.ERROR.value

    def test_cancelled_matches_risky(self):
        assert classify("I want my order cancelled") == Route.RISKY.value

    def test_deleted_matches_risky(self):
        assert classify("My account got deleted") == Route.RISKY.value

    def test_finding_matches_tool(self):
        assert classify("I need help finding my order") == Route.TOOL.value


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_query_defaults_simple(self):
        assert classify("") == Route.SIMPLE.value

    def test_punctuation_stripped(self):
        assert classify("Can you fix it?!") == Route.MISSING_INFO.value

    def test_missing_info_needs_it(self):
        """Short query but no 'it' — should be simple not missing_info."""
        assert classify("Help me please") == Route.SIMPLE.value

    def test_error_system_failure(self):
        assert classify("System failure cannot recover after multiple attempts") == Route.ERROR.value

    def test_risky_send(self):
        assert classify("Send confirmation email to customer") == Route.RISKY.value
