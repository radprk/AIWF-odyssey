"""Tests for persistent memory system."""

from __future__ import annotations

import pytest
from pathlib import Path


class TestConversationStore:
    """Tests for ConversationStore."""

    def test_create_conversation(self, conversation_store):
        """Test creating a new conversation."""
        conv = conversation_store.create_conversation(
            conversation_id="test-001",
            session_id="session-001",
            customer_id="customer-001",
        )

        assert conv.id == "test-001"
        assert conv.session_id == "session-001"
        assert conv.customer_id == "customer-001"
        assert conv.status == "active"
        assert conv.messages == []

    def test_add_message(self, conversation_store):
        """Test adding messages to conversation."""
        conversation_store.create_conversation(
            conversation_id="test-002",
            session_id="session-001",
        )

        msg = conversation_store.add_message(
            conversation_id="test-002",
            role="customer",
            content="Hello, I need help!",
        )

        assert msg.conversation_id == "test-002"
        assert msg.role == "customer"
        assert msg.content == "Hello, I need help!"
        assert msg.id is not None

    def test_get_conversation_with_messages(self, conversation_store):
        """Test retrieving conversation with all messages."""
        conversation_store.create_conversation(
            conversation_id="test-003",
            session_id="session-001",
        )

        conversation_store.add_message("test-003", "customer", "Question")
        conversation_store.add_message("test-003", "agent", "Answer", agent_name="L1_Support")

        conv = conversation_store.get_conversation("test-003")

        assert conv is not None
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "customer"
        assert conv.messages[1].role == "agent"
        assert conv.messages[1].agent_name == "L1_Support"

    def test_get_nonexistent_conversation(self, conversation_store):
        """Test getting a conversation that doesn't exist."""
        conv = conversation_store.get_conversation("nonexistent")
        assert conv is None

    def test_update_conversation_status(self, conversation_store):
        """Test updating conversation status."""
        conversation_store.create_conversation(
            conversation_id="test-004",
            session_id="session-001",
        )

        conversation_store.update_conversation_status(
            "test-004",
            status="resolved",
            resolution_summary="Issue resolved successfully",
        )

        conv = conversation_store.get_conversation("test-004")
        assert conv.status == "resolved"
        assert conv.resolution_summary == "Issue resolved successfully"
        assert conv.ended_at is not None

    def test_get_customer_conversations(self, conversation_store):
        """Test getting all conversations for a customer."""
        # Create multiple conversations
        for i in range(3):
            conversation_store.create_conversation(
                conversation_id=f"conv-{i}",
                session_id=f"session-{i}",
                customer_id="customer-001",
            )

        conversations = conversation_store.get_customer_conversations("customer-001")
        assert len(conversations) == 3

    def test_search_conversations(self, conversation_store):
        """Test searching conversations by content."""
        conversation_store.create_conversation("test-005", "session-001")
        conversation_store.add_message("test-005", "customer", "I have a billing question")

        conversation_store.create_conversation("test-006", "session-002")
        conversation_store.add_message("test-006", "customer", "Password reset needed")

        results = conversation_store.search_conversations("billing")
        assert len(results) == 1
        assert results[0].id == "test-005"

    def test_get_conversation_history_context(self, conversation_store):
        """Test generating context from conversation history."""
        conversation_store.create_conversation(
            conversation_id="test-007",
            session_id="session-001",
            customer_id="customer-001",
        )
        conversation_store.add_message("test-007", "customer", "Previous question")
        conversation_store.add_message("test-007", "agent", "Previous answer", agent_name="L1")
        conversation_store.update_conversation_status(
            "test-007", "resolved", "Issue was resolved"
        )

        context = conversation_store.get_conversation_history_context("customer-001")

        assert "Previous Conversation History" in context
        assert "Previous question" in context
        assert "Previous answer" in context

    def test_statistics(self, conversation_store):
        """Test getting statistics."""
        conversation_store.create_conversation("stat-001", "session-001")
        conversation_store.create_conversation("stat-002", "session-002")
        conversation_store.add_message("stat-001", "customer", "Test")
        conversation_store.update_conversation_status("stat-001", "resolved")

        stats = conversation_store.get_statistics()

        assert stats["total_conversations"] == 2
        assert stats["total_messages"] == 1
        assert "resolved" in stats["by_status"]

    def test_clear_all(self, conversation_store):
        """Test clearing all data."""
        conversation_store.create_conversation("clear-001", "session-001")
        conversation_store.add_message("clear-001", "customer", "Test")

        conversation_store.clear_all()

        stats = conversation_store.get_statistics()
        assert stats["total_conversations"] == 0
        assert stats["total_messages"] == 0


class TestCustomerProfileStore:
    """Tests for CustomerProfileStore."""

    def test_create_customer(self, customer_store):
        """Test creating a new customer."""
        profile = customer_store.create_or_get_customer(
            customer_id="cust-001",
            name="John Doe",
            email="john@example.com",
        )

        assert profile.id == "cust-001"
        assert profile.name == "John Doe"
        assert profile.email == "john@example.com"
        assert profile.total_interactions == 0

    def test_get_existing_customer(self, customer_store):
        """Test getting existing customer doesn't create duplicate."""
        customer_store.create_or_get_customer("cust-002", "Jane")

        profile = customer_store.create_or_get_customer("cust-002", "Different Name")

        assert profile.name == "Jane"  # Original name preserved

    def test_update_customer(self, customer_store):
        """Test updating customer profile."""
        profile = customer_store.create_or_get_customer("cust-003")
        profile.account_tier = "premium"
        profile.tags = ["vip", "business"]

        customer_store.update_customer(profile)

        updated = customer_store.get_customer("cust-003")
        assert updated.account_tier == "premium"
        assert "vip" in updated.tags

    def test_record_interaction(self, customer_store):
        """Test recording interactions updates statistics."""
        customer_store.create_or_get_customer("cust-004")

        customer_store.record_interaction(
            "cust-004",
            was_escalated=True,
            was_resolved=True,
            sentiment=0.5,
        )

        profile = customer_store.get_customer("cust-004")
        assert profile.total_interactions == 1
        assert profile.escalation_count == 1
        assert profile.resolved_count == 1
        assert profile.last_sentiment == 0.5

    def test_sentiment_averaging(self, customer_store):
        """Test that sentiment is properly averaged."""
        customer_store.create_or_get_customer("cust-005")

        customer_store.record_interaction("cust-005", sentiment=1.0)
        customer_store.record_interaction("cust-005", sentiment=-1.0)
        customer_store.record_interaction("cust-005", sentiment=0.0)

        profile = customer_store.get_customer("cust-005")
        assert profile.total_interactions == 3
        assert abs(profile.average_sentiment - 0.0) < 0.1

    def test_add_known_issue(self, customer_store):
        """Test adding known issues."""
        customer_store.create_or_get_customer("cust-006")

        customer_store.add_known_issue("cust-006", "Billing disputes")
        customer_store.add_known_issue("cust-006", "Account access")

        profile = customer_store.get_customer("cust-006")
        assert "Billing disputes" in profile.known_issues
        assert len(profile.known_issues) == 2

    def test_add_tag(self, customer_store):
        """Test adding tags."""
        customer_store.create_or_get_customer("cust-007")

        customer_store.add_tag("cust-007", "premium")
        customer_store.add_tag("cust-007", "premium")  # Duplicate

        profile = customer_store.get_customer("cust-007")
        assert profile.tags.count("premium") == 1

    def test_search_by_tag(self, customer_store):
        """Test searching customers by tag."""
        customer_store.create_or_get_customer("cust-008")
        customer_store.add_tag("cust-008", "business")

        customer_store.create_or_get_customer("cust-009")
        customer_store.add_tag("cust-009", "personal")

        results = customer_store.search_by_tag("business")
        assert len(results) == 1
        assert results[0].id == "cust-008"

    def test_get_high_escalation_customers(self, customer_store):
        """Test finding high escalation customers."""
        customer_store.create_or_get_customer("cust-010")
        for _ in range(5):
            customer_store.record_interaction("cust-010", was_escalated=True)

        customer_store.create_or_get_customer("cust-011")
        for _ in range(5):
            customer_store.record_interaction("cust-011", was_escalated=False)

        high_escalation = customer_store.get_high_escalation_customers(threshold=0.3)
        assert len(high_escalation) == 1
        assert high_escalation[0].id == "cust-010"

    def test_context_summary(self, customer_store):
        """Test generating context summary for agents."""
        profile = customer_store.create_or_get_customer("cust-012", "Alice Smith")
        profile.account_tier = "vip"
        profile.last_sentiment = -0.5
        profile.known_issues = ["Password resets", "Fee disputes"]
        profile.tags = ["priority"]
        customer_store.update_customer(profile)

        summary = profile.get_context_summary()

        assert "Alice Smith" in summary
        assert "VIP" in summary
        assert "negative" in summary.lower()
        assert "Password resets" in summary


class TestSessionManager:
    """Tests for SessionManager."""

    def test_create_session(self, session_manager):
        """Test creating a new session."""
        state = session_manager.create_session(
            session_id="sess-001",
            customer_id="cust-001",
        )

        assert state.session_id == "sess-001"
        assert state.customer_id == "cust-001"
        assert state.conversation_id is None

    def test_start_conversation(self, session_manager):
        """Test starting a conversation within session."""
        session_manager.create_session("sess-002", "cust-002")

        conv = session_manager.start_conversation(
            "sess-002",
            "Hello, I need help with my account",
        )

        assert conv.session_id == "sess-002"
        assert conv.customer_id == "cust-002"
        assert len(conv.messages) == 0  # Message added separately

    def test_add_messages(self, session_manager):
        """Test adding messages to conversation."""
        session_manager.create_session("sess-003")
        session_manager.start_conversation("sess-003", "Initial query")

        session_manager.add_agent_response(
            "sess-003",
            agent_name="L1_Support",
            response="How can I help?",
        )

        state = session_manager.get_session("sess-003")
        assert state.current_agent == "L1_Support"

    def test_record_escalation(self, session_manager):
        """Test recording escalation."""
        session_manager.create_session("sess-004")
        session_manager.start_conversation("sess-004", "Complex issue")

        session_manager.record_escalation(
            "sess-004",
            from_agent="L1_Support",
            to_agent="L2_Support",
            reason="Technical issue",
        )

        state = session_manager.get_session("sess-004")
        assert "L1_Support -> L2_Support" in state.escalation_chain
        assert state.current_agent == "L2_Support"

    def test_resolve_conversation(self, session_manager, customer_store):
        """Test resolving conversation updates customer profile."""
        session_manager.create_session("sess-005", "cust-005")
        session_manager.start_conversation("sess-005", "Issue to resolve")

        session_manager.resolve_conversation(
            "sess-005",
            resolution_summary="Issue resolved successfully",
            sentiment=0.8,
        )

        # Check customer profile was updated
        profile = customer_store.get_customer("cust-005")
        assert profile.resolved_count == 1
        assert profile.last_sentiment == 0.8

    def test_get_context_for_agent(self, session_manager, customer_store):
        """Test generating full context for agent."""
        # Set up customer with history
        profile = customer_store.create_or_get_customer("cust-006", "Bob")
        profile.account_tier = "premium"
        customer_store.update_customer(profile)

        session_manager.create_session("sess-006", "cust-006")
        session_manager.start_conversation("sess-006", "New question")
        session_manager.add_agent_response("sess-006", "L1", "Response 1")

        context = session_manager.get_context_for_agent("sess-006")

        assert "Bob" in context or "cust-006" in context

    def test_end_session(self, session_manager):
        """Test ending a session."""
        session_manager.create_session("sess-007")
        session_manager.start_conversation("sess-007", "Test")

        summary = session_manager.end_session("sess-007")

        assert summary["session_id"] == "sess-007"
        assert session_manager.get_session("sess-007") is None

    def test_multiple_active_sessions(self, session_manager):
        """Test managing multiple active sessions."""
        session_manager.create_session("sess-008")
        session_manager.create_session("sess-009")
        session_manager.create_session("sess-010")

        active = session_manager.get_active_sessions()
        assert len(active) == 3
