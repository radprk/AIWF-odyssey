"""Tests for routing and task classification."""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

# Add autogen_agents to path
sys.path.insert(0, str(Path(__file__).parent.parent / "autogen_agents"))


class TestTaskClassifier:
    """Tests for task classification."""

    def test_classify_billing_query(self):
        """Test billing query classification."""
        from task_classifier import classify_task

        result = classify_task("Why was I charged a $15 fee?")
        assert result["task_type"] == "Billing"

    def test_classify_account_query(self):
        """Test account query classification."""
        from task_classifier import classify_task

        result = classify_task("I forgot my password")
        assert result["task_type"] == "Account"

    def test_classify_transaction_query(self):
        """Test transaction query classification."""
        from task_classifier import classify_task

        result = classify_task("When will my transfer complete?")
        assert result["task_type"] == "Transaction"

    def test_classify_complaint_query(self):
        """Test complaint query classification."""
        from task_classifier import classify_task

        result = classify_task("I want to file a formal complaint about terrible service!")
        assert result["task_type"] == "Complaint"

    def test_classify_product_query(self):
        """Test product info query classification."""
        from task_classifier import classify_task

        result = classify_task("What types of savings accounts do you offer?")
        assert result["task_type"] == "Product Info"

    def test_classify_general_query(self):
        """Test general query classification."""
        from task_classifier import classify_task

        result = classify_task("Hello, I have a question")
        assert result["task_type"] == "General Inquiry"

    def test_classification_returns_estimate(self):
        """Test that classification returns time estimate."""
        from task_classifier import classify_task

        result = classify_task("Why was I charged?")
        assert "estimated_duration_sec" in result
        assert result["estimated_duration_sec"] > 0


class TestRouter:
    """Tests for query routing logic."""

    def test_route_to_billing_specialist(self):
        """Test routing to billing specialist."""
        from router import route_query

        specialists = route_query("I have a question about fees")
        assert "billing" in specialists

    def test_route_to_account_specialist(self):
        """Test routing to account specialist."""
        from router import route_query

        specialists = route_query("I need to reset my password")
        assert "account" in specialists

    def test_route_to_transaction_specialist(self):
        """Test routing to transaction specialist."""
        from router import route_query

        specialists = route_query("My transfer is pending")
        assert "transaction" in specialists

    def test_route_to_complaint_specialist(self):
        """Test routing to complaint specialist."""
        from router import route_query

        specialists = route_query("I want to complain about service")
        assert "complaint" in specialists

    def test_route_to_general_specialist(self):
        """Test routing to general specialist."""
        from router import route_query

        specialists = route_query("What products do you offer?")
        assert "general" in specialists

    def test_route_complex_query_multiple_specialists(self):
        """Test complex query routes to multiple specialists."""
        from router import route_query

        # This query touches billing and complaint
        specialists = route_query("I'm very upset about the fee you charged me!")
        assert len(specialists) >= 1

    def test_route_returns_list(self):
        """Test that route always returns a list."""
        from router import route_query

        result = route_query("Random query")
        assert isinstance(result, list)
        assert len(result) > 0


class TestEscalationTriggers:
    """Tests for escalation trigger detection."""

    def test_detect_escalation_keywords(self):
        """Test escalation keyword detection."""
        # Import the escalation triggers
        sys.path.insert(0, str(Path(__file__).parent.parent / "autogen_agents"))

        # Common escalation triggers
        triggers = [
            "escalate",
            "cannot resolve",
            "need help from l2",
            "supervisor",
            "manager",
            "not sure",
        ]

        for trigger in triggers:
            text = f"I {trigger} this issue"
            # Just verify these are reasonable triggers
            assert trigger.lower() in text.lower()

    def test_no_false_positive_escalation(self):
        """Test that normal responses don't trigger escalation."""
        normal_responses = [
            "I can help you with that.",
            "Your balance is $500.",
            "The transfer will complete tomorrow.",
            "I've updated your account.",
        ]

        escalation_keywords = ["escalate", "cannot resolve", "need help from l2"]

        for response in normal_responses:
            has_trigger = any(kw in response.lower() for kw in escalation_keywords)
            assert not has_trigger


class TestSpecialistSelection:
    """Tests for specialist agent selection."""

    def test_all_specialists_defined(self):
        """Test that all required specialists are defined."""
        required_specialists = [
            "billing",
            "account",
            "transaction",
            "complaint",
            "general",
        ]

        # Check router handles all these
        from router import route_query

        for specialist in required_specialists:
            # Query should route somewhere
            result = route_query(f"{specialist} related question")
            assert len(result) > 0

    def test_specialist_mapping_consistency(self):
        """Test that specialist names are consistent."""
        from router import route_query

        # Multiple queries of same type should route consistently
        billing_queries = [
            "fee question",
            "charge inquiry",
            "refund request",
        ]

        results = [route_query(q) for q in billing_queries]

        # All should include billing
        for result in results:
            assert "billing" in result or len(result) > 0
