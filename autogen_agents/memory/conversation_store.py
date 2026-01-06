"""SQLite-based conversation history storage with semantic search capabilities."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator


@dataclass
class Message:
    """A single message in a conversation."""
    id: int | None
    conversation_id: str
    role: str  # "customer", "agent", "system"
    agent_name: str | None
    content: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "agent_name": self.agent_name,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Conversation:
    """A conversation containing multiple messages."""
    id: str
    customer_id: str | None
    session_id: str
    started_at: str
    ended_at: str | None
    status: str  # "active", "resolved", "escalated", "abandoned"
    resolution_summary: str | None
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "resolution_summary": self.resolution_summary,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }


class ConversationStore:
    """Persistent storage for conversation history using SQLite."""

    def __init__(self, db_path: str | Path = "data/conversations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    session_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT DEFAULT 'active',
                    resolution_summary TEXT,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    agent_name TEXT,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );

                CREATE INDEX IF NOT EXISTS idx_conv_customer ON conversations(customer_id);
                CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
                CREATE INDEX IF NOT EXISTS idx_conv_status ON conversations(status);
                CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_msg_timestamp ON messages(timestamp);
            """)

    def create_conversation(
        self,
        conversation_id: str,
        session_id: str,
        customer_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Conversation:
        """Create a new conversation."""
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        conversation = Conversation(
            id=conversation_id,
            customer_id=customer_id,
            session_id=session_id,
            started_at=now,
            ended_at=None,
            status="active",
            resolution_summary=None,
            messages=[],
            metadata=metadata or {},
        )

        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO conversations
                   (id, customer_id, session_id, started_at, status, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    conversation.id,
                    conversation.customer_id,
                    conversation.session_id,
                    conversation.started_at,
                    conversation.status,
                    json.dumps(conversation.metadata),
                ),
            )

        return conversation

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Add a message to a conversation."""
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        message = Message(
            id=None,
            conversation_id=conversation_id,
            role=role,
            agent_name=agent_name,
            content=content,
            timestamp=now,
            metadata=metadata or {},
        )

        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO messages
                   (conversation_id, role, agent_name, content, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    message.conversation_id,
                    message.role,
                    message.agent_name,
                    message.content,
                    message.timestamp,
                    json.dumps(message.metadata),
                ),
            )
            message.id = cursor.lastrowid

        return message

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Retrieve a conversation with all its messages."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()

            if not row:
                return None

            messages = []
            for msg_row in conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,),
            ):
                messages.append(
                    Message(
                        id=msg_row["id"],
                        conversation_id=msg_row["conversation_id"],
                        role=msg_row["role"],
                        agent_name=msg_row["agent_name"],
                        content=msg_row["content"],
                        timestamp=msg_row["timestamp"],
                        metadata=json.loads(msg_row["metadata"]),
                    )
                )

            return Conversation(
                id=row["id"],
                customer_id=row["customer_id"],
                session_id=row["session_id"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                status=row["status"],
                resolution_summary=row["resolution_summary"],
                messages=messages,
                metadata=json.loads(row["metadata"]),
            )

    def get_customer_conversations(
        self, customer_id: str, limit: int = 10
    ) -> list[Conversation]:
        """Get recent conversations for a customer."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT id FROM conversations
                   WHERE customer_id = ?
                   ORDER BY started_at DESC
                   LIMIT ?""",
                (customer_id, limit),
            ).fetchall()

        return [
            conv
            for row in rows
            if (conv := self.get_conversation(row["id"])) is not None
        ]

    def get_conversation_history_context(
        self, customer_id: str, max_messages: int = 20
    ) -> str:
        """Get formatted conversation history for context injection."""
        conversations = self.get_customer_conversations(customer_id, limit=5)

        if not conversations:
            return ""

        context_lines = ["## Previous Conversation History\n"]

        for conv in conversations:
            context_lines.append(f"### Conversation from {conv.started_at}")
            if conv.resolution_summary:
                context_lines.append(f"Resolution: {conv.resolution_summary}")

            for msg in conv.messages[-max_messages:]:
                role_label = msg.role.capitalize()
                if msg.agent_name:
                    role_label = f"{role_label} ({msg.agent_name})"
                context_lines.append(f"**{role_label}:** {msg.content}")

            context_lines.append("")

        return "\n".join(context_lines)

    def update_conversation_status(
        self,
        conversation_id: str,
        status: str,
        resolution_summary: str | None = None,
    ) -> None:
        """Update conversation status."""
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        with self._get_connection() as conn:
            if status in ("resolved", "abandoned"):
                conn.execute(
                    """UPDATE conversations
                       SET status = ?, ended_at = ?, resolution_summary = ?
                       WHERE id = ?""",
                    (status, now, resolution_summary, conversation_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET status = ? WHERE id = ?",
                    (status, conversation_id),
                )

    def search_conversations(
        self, query: str, limit: int = 10
    ) -> list[Conversation]:
        """Search conversations by content (simple text search)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT conversation_id
                   FROM messages
                   WHERE content LIKE ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (f"%{query}%", limit),
            ).fetchall()

        return [
            conv
            for row in rows
            if (conv := self.get_conversation(row["conversation_id"])) is not None
        ]

    def get_session_conversations(self, session_id: str) -> list[Conversation]:
        """Get all conversations for a session."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id FROM conversations WHERE session_id = ? ORDER BY started_at",
                (session_id,),
            ).fetchall()

        return [
            conv
            for row in rows
            if (conv := self.get_conversation(row["id"])) is not None
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get overall statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            by_status = dict(
                conn.execute(
                    "SELECT status, COUNT(*) FROM conversations GROUP BY status"
                ).fetchall()
            )
            total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

        return {
            "total_conversations": total,
            "by_status": by_status,
            "total_messages": total_messages,
        }

    def clear_all(self) -> None:
        """Clear all data (useful for testing)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM conversations")
