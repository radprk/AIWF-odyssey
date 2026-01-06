"""Integration tests for the multi-agent system.

These tests require Ollama to be running with the required models.
Mark with @pytest.mark.llm to skip when Ollama is unavailable.
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

# Add autogen_agents to path
sys.path.insert(0, str(Path(__file__).parent.parent / "autogen_agents"))


def ollama_available() -> bool:
    """Check if Ollama is available."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


# Skip marker for tests requiring Ollama
requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama not available"
)


class TestMemoryIntegration:
    """Integration tests for memory system."""

    def test_full_conversation_flow(self, session_manager):
        """Test complete conversation flow with memory."""
        # Create session
        session = session_manager.create_session(
            session_id="integration-001",
            customer_id="cust-integration",
        )

        # Start conversation
        conv = session_manager.start_conversation(
            "integration-001",
            "I have a billing question",
        )

        # Simulate agent responses
        session_manager.add_agent_response(
            "integration-001",
            "L1_Support",
            "How can I help with billing?",
        )

        session_manager.add_customer_message(
            "integration-001",
            "Why was I charged $15?",
        )

        # Escalate
        session_manager.record_escalation(
            "integration-001",
            "L1_Support",
            "L2_Support",
            "Complex billing issue",
        )

        session_manager.add_agent_response(
            "integration-001",
            "L2_Support",
            "I can see this was a maintenance fee. Let me waive it for you.",
        )

        # Resolve
        session_manager.resolve_conversation(
            "integration-001",
            "Fee waived successfully",
            sentiment=0.8,
        )

        # Verify conversation stored
        conv = session_manager.conversation_store.get_conversation(
            session.conversation_id
        )
        assert conv is not None
        assert conv.status == "resolved"

        # Verify customer profile updated
        profile = session_manager.customer_store.get_customer("cust-integration")
        assert profile.total_interactions == 1
        assert profile.escalation_count == 1
        assert profile.resolved_count == 1

    def test_context_retrieval_for_returning_customer(self, session_manager):
        """Test that returning customers get context from history."""
        customer_id = "returning-customer"

        # First session
        session_manager.create_session("sess-1", customer_id)
        session_manager.start_conversation("sess-1", "First question about fees")
        session_manager.add_agent_response("sess-1", "L1", "First response")
        session_manager.resolve_conversation("sess-1", "First issue resolved")

        # Second session - should have history
        session_manager.create_session("sess-2", customer_id)
        session_manager.start_conversation("sess-2", "New question")

        context = session_manager.get_context_for_agent("sess-2")

        # Should include previous conversation
        assert "Previous Conversation" in context or "First" in context


class TestBenchmarkIntegration:
    """Integration tests for benchmark system."""

    def test_benchmark_suite_save_load_roundtrip(self, temp_dir, benchmark_suite):
        """Test saving and loading benchmark suite preserves data."""
        path = temp_dir / "benchmarks.json"
        benchmark_suite.save(path)

        from evaluation.benchmarks import BenchmarkSuite
        loaded = BenchmarkSuite.load(path)

        assert loaded.name == benchmark_suite.name
        assert len(loaded.cases) == len(benchmark_suite.cases)

        # Check individual cases preserved
        for orig, loaded_case in zip(benchmark_suite.cases, loaded.cases):
            assert orig.id == loaded_case.id
            assert orig.query == loaded_case.query
            assert orig.should_escalate == loaded_case.should_escalate

    def test_benchmark_filtering(self, benchmark_suite):
        """Test filtering benchmark cases."""
        billing = benchmark_suite.filter_by_category("billing")
        assert all(c.category == "billing" for c in billing)

        easy = benchmark_suite.filter_by_difficulty("easy")
        assert all(c.difficulty == "easy" for c in easy)

    def test_report_generation_with_mock_results(self):
        """Test report generation with simulated results."""
        from evaluation.benchmarks import BenchmarkCase, BenchmarkResult
        from evaluation.metrics import EvaluationResult
        from evaluation.reports import generate_evaluation_report

        # Create diverse test cases
        cases = [
            BenchmarkCase("1", "Easy billing", "billing", "easy", "Q1"),
            BenchmarkCase("2", "Medium account", "account", "medium", "Q2"),
            BenchmarkCase("3", "Hard transaction", "transaction", "hard", "Q3"),
        ]

        # Simulate results
        results = []
        for i, case in enumerate(cases):
            eval_result = EvaluationResult(
                relevance=4.0 - i * 0.5,
                faithfulness=4.0 - i * 0.3,
                helpfulness=4.0 - i * 0.4,
                tone_appropriateness=4.0,
                completeness=3.5,
                accuracy=4.0,
                clarity=4.0,
                actionability=3.5,
                processing_time_ms=100 + i * 50,
            )

            results.append(BenchmarkResult(
                case=case,
                response=f"Response {i}",
                evaluation=eval_result,
                actual_specialists=[case.category],
                was_escalated=i == 2,  # Hard case escalated
                execution_time_ms=500 + i * 100,
                timestamp="2024-01-01T00:00:00Z",
            ))

        report = generate_evaluation_report(results, "Integration Test")

        # Verify report structure
        assert report.total_cases == 3
        assert "billing" in report.by_category
        assert "easy" in report.by_difficulty
        assert report.avg_execution_time_ms > 0


@requires_ollama
class TestLLMIntegration:
    """Integration tests requiring LLM (Ollama)."""

    def test_response_evaluator_basic(self):
        """Test ResponseEvaluator with real LLM."""
        from evaluation.metrics import ResponseEvaluator

        evaluator = ResponseEvaluator()

        result = evaluator.evaluate(
            query="Why was I charged a fee?",
            response="I understand you're concerned about the fee. This is a monthly maintenance fee that applies to accounts below the minimum balance. I'd be happy to review your account and see if we can waive it for you.",
            context="Monthly maintenance fees are $15 for accounts below $500 minimum balance.",
        )

        # Should get reasonable scores
        assert result.relevance > 0
        assert result.helpfulness > 0
        assert result.overall_score > 0

    def test_response_evaluator_detects_poor_response(self):
        """Test that evaluator detects poor responses."""
        from evaluation.metrics import ResponseEvaluator

        evaluator = ResponseEvaluator()

        result = evaluator.evaluate(
            query="Why was I charged a fee?",
            response="idk",
            context="Monthly maintenance fees are $15 for accounts below $500 minimum balance.",
        )

        # Poor response should score low
        assert result.overall_score < 3.0

    def test_task_classifier_integration(self):
        """Test task classifier with various queries."""
        from task_classifier import classify_task

        test_cases = [
            ("Why am I being charged?", "Billing"),
            ("I can't log in", "Account"),
            ("Transfer money to savings", "Transaction"),
            ("This is terrible service!", "Complaint"),
            ("What products do you offer?", "Product Info"),
        ]

        for query, expected_type in test_cases:
            result = classify_task(query)
            assert result["task_type"] == expected_type, f"Failed for: {query}"

    def test_end_to_end_query_handling(self):
        """Test complete query handling flow."""
        from simulation_engine import simulate_query_handling

        response = simulate_query_handling(
            qid=1,
            query="Why was I charged a $15 fee on my account?",
            session_id="e2e-test",
            use_router=True,
            use_llm_router=False,
        )

        # Should get a non-empty response
        assert response is not None
        assert len(response) > 0


class TestDataPersistence:
    """Tests for data persistence across sessions."""

    def test_conversation_persists_after_store_recreation(self, temp_db_path):
        """Test conversations persist after store is recreated."""
        from memory.conversation_store import ConversationStore

        # Create store and add data
        store1 = ConversationStore(db_path=temp_db_path)
        store1.create_conversation("persist-test", "session-1", "customer-1")
        store1.add_message("persist-test", "customer", "Test message")

        # Create new store instance pointing to same DB
        store2 = ConversationStore(db_path=temp_db_path)

        # Data should be there
        conv = store2.get_conversation("persist-test")
        assert conv is not None
        assert len(conv.messages) == 1
        assert conv.messages[0].content == "Test message"

    def test_customer_profile_persists(self, temp_db_path):
        """Test customer profiles persist."""
        from memory.customer_profile import CustomerProfileStore

        db_path = temp_db_path.parent / "customers_persist.db"

        # Create and update profile
        store1 = CustomerProfileStore(db_path=db_path)
        store1.create_or_get_customer("persist-cust", "John Doe")
        store1.record_interaction("persist-cust", was_escalated=True, sentiment=-0.5)
        store1.add_tag("persist-cust", "vip")

        # Recreate store
        store2 = CustomerProfileStore(db_path=db_path)
        profile = store2.get_customer("persist-cust")

        assert profile.name == "John Doe"
        assert profile.total_interactions == 1
        assert profile.escalation_count == 1
        assert profile.last_sentiment == -0.5
        assert "vip" in profile.tags


class TestErrorHandling:
    """Tests for error handling."""

    def test_get_nonexistent_conversation(self, conversation_store):
        """Test getting non-existent conversation returns None."""
        result = conversation_store.get_conversation("does-not-exist")
        assert result is None

    def test_get_nonexistent_customer(self, customer_store):
        """Test getting non-existent customer returns None."""
        result = customer_store.get_customer("does-not-exist")
        assert result is None

    def test_add_message_to_nonexistent_conversation(self, conversation_store):
        """Test adding message to non-existent conversation."""
        # This should not raise, but message won't be retrievable
        # without a parent conversation
        msg = conversation_store.add_message(
            "nonexistent-conv",
            "customer",
            "Test",
        )
        # Message is created but orphaned
        assert msg.content == "Test"

    def test_session_manager_handles_missing_session(self, session_manager):
        """Test session manager handles missing session gracefully."""
        # These should not raise
        session_manager.add_agent_response("missing", "Agent", "Response")
        session_manager.add_customer_message("missing", "Message")

        context = session_manager.get_context_for_agent("missing")
        assert context == ""

        summary = session_manager.end_session("missing")
        assert summary == {}
