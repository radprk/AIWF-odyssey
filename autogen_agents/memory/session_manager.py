"""Session management for multi-agent conversations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .conversation_store import ConversationStore, Conversation
from .customer_profile import CustomerProfileStore, CustomerProfile


@dataclass
class SessionState:
    """Current state of an active session."""
    session_id: str
    customer_id: str | None
    conversation_id: str | None
    started_at: str

    # Current agent handling the session
    current_agent: str | None = None

    # Escalation chain
    escalation_chain: list[str] = field(default_factory=list)

    # Accumulated context
    context: dict[str, Any] = field(default_factory=dict)

    # Session metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Manages session state and coordinates memory stores."""

    def __init__(
        self,
        conversation_store: ConversationStore | None = None,
        customer_store: CustomerProfileStore | None = None,
    ):
        self.conversation_store = conversation_store or ConversationStore()
        self.customer_store = customer_store or CustomerProfileStore()
        self._active_sessions: dict[str, SessionState] = {}

    def create_session(
        self,
        session_id: str | None = None,
        customer_id: str | None = None,
    ) -> SessionState:
        """Create a new session."""
        session_id = session_id or datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        # Create or get customer profile
        if customer_id:
            self.customer_store.create_or_get_customer(customer_id)

        state = SessionState(
            session_id=session_id,
            customer_id=customer_id,
            conversation_id=None,
            started_at=now,
        )

        self._active_sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> SessionState | None:
        """Get an active session."""
        return self._active_sessions.get(session_id)

    def start_conversation(
        self,
        session_id: str,
        initial_query: str,
        metadata: dict[str, Any] | None = None,
    ) -> Conversation:
        """Start a new conversation within a session."""
        state = self._active_sessions.get(session_id)
        if not state:
            state = self.create_session(session_id)

        # Generate conversation ID
        conv_id = f"{session_id}_conv_{uuid.uuid4().hex[:8]}"

        # Create conversation in store
        conversation = self.conversation_store.create_conversation(
            conversation_id=conv_id,
            session_id=session_id,
            customer_id=state.customer_id,
            metadata=metadata,
        )

        # Add initial customer message
        self.conversation_store.add_message(
            conversation_id=conv_id,
            role="customer",
            content=initial_query,
        )

        state.conversation_id = conv_id
        return conversation

    def add_agent_response(
        self,
        session_id: str,
        agent_name: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an agent response to the current conversation."""
        state = self._active_sessions.get(session_id)
        if not state or not state.conversation_id:
            return

        self.conversation_store.add_message(
            conversation_id=state.conversation_id,
            role="agent",
            agent_name=agent_name,
            content=response,
            metadata=metadata,
        )

        state.current_agent = agent_name

    def add_customer_message(
        self,
        session_id: str,
        message: str,
    ) -> None:
        """Add a customer message to the current conversation."""
        state = self._active_sessions.get(session_id)
        if not state or not state.conversation_id:
            return

        self.conversation_store.add_message(
            conversation_id=state.conversation_id,
            role="customer",
            content=message,
        )

    def record_escalation(
        self,
        session_id: str,
        from_agent: str,
        to_agent: str,
        reason: str | None = None,
    ) -> None:
        """Record an escalation event."""
        state = self._active_sessions.get(session_id)
        if not state:
            return

        state.escalation_chain.append(f"{from_agent} -> {to_agent}")
        state.current_agent = to_agent

        if state.conversation_id:
            self.conversation_store.add_message(
                conversation_id=state.conversation_id,
                role="system",
                content=f"Escalated from {from_agent} to {to_agent}" + (f": {reason}" if reason else ""),
                metadata={"type": "escalation", "from": from_agent, "to": to_agent},
            )
            self.conversation_store.update_conversation_status(
                state.conversation_id, "escalated"
            )

    def resolve_conversation(
        self,
        session_id: str,
        resolution_summary: str,
        sentiment: float | None = None,
    ) -> None:
        """Mark current conversation as resolved."""
        state = self._active_sessions.get(session_id)
        if not state or not state.conversation_id:
            return

        self.conversation_store.update_conversation_status(
            state.conversation_id,
            "resolved",
            resolution_summary=resolution_summary,
        )

        # Update customer profile
        if state.customer_id:
            was_escalated = len(state.escalation_chain) > 0
            self.customer_store.record_interaction(
                customer_id=state.customer_id,
                was_escalated=was_escalated,
                was_resolved=True,
                sentiment=sentiment,
            )

    def get_context_for_agent(self, session_id: str) -> str:
        """Get full context for agent prompt injection."""
        state = self._active_sessions.get(session_id)
        if not state:
            return ""

        context_parts = []

        # Add customer profile context
        if state.customer_id:
            profile = self.customer_store.get_customer(state.customer_id)
            if profile:
                context_parts.append(profile.get_context_summary())

                # Add previous conversation history
                history = self.conversation_store.get_conversation_history_context(
                    state.customer_id, max_messages=10
                )
                if history:
                    context_parts.append(history)

        # Add current conversation context
        if state.conversation_id:
            conv = self.conversation_store.get_conversation(state.conversation_id)
            if conv and conv.messages:
                context_parts.append("## Current Conversation")
                for msg in conv.messages[-5:]:  # Last 5 messages
                    role = msg.role.capitalize()
                    if msg.agent_name:
                        role = f"{role} ({msg.agent_name})"
                    context_parts.append(f"**{role}:** {msg.content}")

        # Add escalation context
        if state.escalation_chain:
            context_parts.append(f"\n## Escalation History: {' -> '.join(state.escalation_chain)}")

        return "\n\n".join(context_parts)

    def end_session(self, session_id: str) -> dict[str, Any]:
        """End a session and return summary."""
        state = self._active_sessions.pop(session_id, None)
        if not state:
            return {}

        # Mark conversation as abandoned if not resolved
        if state.conversation_id:
            conv = self.conversation_store.get_conversation(state.conversation_id)
            if conv and conv.status == "active":
                self.conversation_store.update_conversation_status(
                    state.conversation_id, "abandoned"
                )

        return {
            "session_id": session_id,
            "customer_id": state.customer_id,
            "conversation_id": state.conversation_id,
            "escalation_chain": state.escalation_chain,
            "duration_started": state.started_at,
        }

    def get_active_sessions(self) -> list[SessionState]:
        """Get all active sessions."""
        return list(self._active_sessions.values())
