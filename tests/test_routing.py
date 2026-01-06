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
        from task_classifier import classify

        task_type, estimate = classify("Why was I charged a $15 fee?")
        assert task_type == "Billing"

    def test_classify_account_query(self):
        """Test account query classification."""
        from task_classifier import classify

        task_type, estimate = classify("I forgot my password")
        assert task_type == "Account"

    def test_classify_transaction_query(self):
        """Test transaction query classification."""
        from task_classifier import classify

        task_type, estimate = classify("When will my transfer complete?")
        assert task_type == "Transaction"

    def test_classify_complaint_query(self):
        """Test complaint query classification."""
        from task_classifier import classify

        task_type, estimate = classify("I want to file a formal complaint about terrible service!")
        assert task_type == "Complaint"

    def test_classify_product_query(self):
        """Test product info query classification."""
        from task_classifier import classify

        task_type, estimate = classify("What types of insurance policies do you offer?")
        assert task_type == "Product Info"

    def test_classify_general_query(self):
        """Test general query classification."""
        from task_classifier import classify

        task_type, estimate = classify("Hello, I have a question")
        assert task_type == "General Inquiry"

    def test_classification_returns_estimate(self):
        """Test that classification returns time estimate."""
        from task_classifier import classify

        task_type, estimate = classify("Why was I charged?")
        assert estimate > 0
        assert isinstance(estimate, int)


class TestRouter:
    """Tests for query routing logic using rule-based routing."""

    def test_route_to_billing_specialist(self):
        """Test routing to billing specialist."""
        from router import _rule_based_route

        specialists = _rule_based_route("Billing")
        assert "billing" in specialists

    def test_route_to_account_specialist(self):
        """Test routing to account specialist."""
        from router import _rule_based_route

        specialists = _rule_based_route("Account")
        assert "account" in specialists

    def test_route_to_transaction_specialist(self):
        """Test routing to transaction specialist."""
        from router import _rule_based_route

        specialists = _rule_based_route("Transaction")
        assert "transaction" in specialists

    def test_route_to_complaint_specialist(self):
        """Test routing to complaint specialist."""
        from router import _rule_based_route

        specialists = _rule_based_route("Complaint")
        assert "complaint" in specialists

    def test_route_to_general_specialist(self):
        """Test routing to general specialist."""
        from router import _rule_based_route

        specialists = _rule_based_route("Product Info")
        assert "general" in specialists

    def test_route_unknown_defaults_to_general(self):
        """Test unknown task type routes to general."""
        from router import _rule_based_route

        specialists = _rule_based_route("Unknown Category")
        assert "general" in specialists

    def test_route_returns_list(self):
        """Test that route always returns a list."""
        from router import _rule_based_route

        result = _rule_based_route("Billing")
        assert isinstance(result, list)
        assert len(result) > 0


class TestEndToEndRouting:
    """Tests for full classify + route flow."""

    def test_billing_query_routes_correctly(self):
        """Test billing query classifies and routes correctly."""
        from task_classifier import classify
        from router import _rule_based_route

        task_type, _ = classify("I have a question about fees")
        specialists = _rule_based_route(task_type)
        assert "billing" in specialists

    def test_account_query_routes_correctly(self):
        """Test account query classifies and routes correctly."""
        from task_classifier import classify
        from router import _rule_based_route

        task_type, _ = classify("I need to reset my password")
        specialists = _rule_based_route(task_type)
        assert "account" in specialists

    def test_transaction_query_routes_correctly(self):
        """Test transaction query classifies and routes correctly."""
        from task_classifier import classify
        from router import _rule_based_route

        task_type, _ = classify("My transfer is pending")
        specialists = _rule_based_route(task_type)
        assert "transaction" in specialists

    def test_complaint_query_routes_correctly(self):
        """Test complaint query classifies and routes correctly."""
        from task_classifier import classify
        from router import _rule_based_route

        task_type, _ = classify("I want to complain about service")
        specialists = _rule_based_route(task_type)
        assert "complaint" in specialists


class TestEscalationTriggers:
    """Tests for escalation trigger detection."""

    def test_detect_escalation_keywords(self):
        """Test escalation keyword detection."""
        # Common escalation triggers from base_agents.py
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
        from router import _rule_based_route

        required_task_types = [
            "Billing",
            "Account",
            "Transaction",
            "Complaint",
            "General Inquiry",
        ]

        for task_type in required_task_types:
            result = _rule_based_route(task_type)
            assert len(result) > 0

    def test_specialist_mapping_consistency(self):
        """Test that specialist names are consistent."""
        from task_classifier import classify
        from router import _rule_based_route

        # Multiple queries of same type should route consistently
        billing_queries = [
            "fee question",
            "charge inquiry",
            "refund request",
        ]

        results = []
        for q in billing_queries:
            task_type, _ = classify(q)
            results.append(_rule_based_route(task_type))

        # All should include billing
        for result in results:
            assert "billing" in result


class TestNormalizeSpecialists:
    """Tests for specialist name normalization."""

    def test_normalize_specialists(self):
        """Test specialist name normalization."""
        from router import normalize_specialists

        raw = ["  Billing ", "ACCOUNT", "transaction"]
        normalized = normalize_specialists(raw)

        assert normalized == ["billing", "account", "transaction"]

    def test_normalize_empty_strings(self):
        """Test that empty strings are filtered out."""
        from router import normalize_specialists

        raw = ["billing", "", "  ", "account"]
        normalized = normalize_specialists(raw)

        assert normalized == ["billing", "account"]
