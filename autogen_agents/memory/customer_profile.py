"""Customer profile persistence for personalized support."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator


@dataclass
class CustomerProfile:
    """Customer profile with history and preferences."""
    id: str
    name: str | None
    email: str | None
    created_at: str
    updated_at: str

    # Interaction statistics
    total_interactions: int = 0
    escalation_count: int = 0
    resolved_count: int = 0

    # Sentiment tracking
    average_sentiment: float = 0.0  # -1 to 1
    last_sentiment: float = 0.0

    # Preferences and context
    preferred_language: str = "en"
    account_tier: str = "standard"  # "standard", "premium", "vip"
    known_issues: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)

    # Tags for categorization
    tags: list[str] = field(default_factory=list)

    # Freeform notes
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_interactions": self.total_interactions,
            "escalation_count": self.escalation_count,
            "resolved_count": self.resolved_count,
            "average_sentiment": self.average_sentiment,
            "last_sentiment": self.last_sentiment,
            "preferred_language": self.preferred_language,
            "account_tier": self.account_tier,
            "known_issues": self.known_issues,
            "preferences": self.preferences,
            "tags": self.tags,
            "notes": self.notes,
        }

    def get_context_summary(self) -> str:
        """Generate a summary for agent context injection."""
        lines = [f"## Customer Profile: {self.id}"]

        if self.name:
            lines.append(f"Name: {self.name}")
        if self.account_tier != "standard":
            lines.append(f"Account Tier: {self.account_tier.upper()}")

        lines.append(f"Total Interactions: {self.total_interactions}")

        if self.escalation_count > 0:
            escalation_rate = self.escalation_count / max(self.total_interactions, 1)
            lines.append(f"Escalation Rate: {escalation_rate:.1%}")

        if self.last_sentiment < -0.3:
            lines.append("⚠️ Customer sentiment is negative - handle with care")
        elif self.last_sentiment > 0.3:
            lines.append("Customer sentiment is positive")

        if self.known_issues:
            lines.append(f"Known Issues: {', '.join(self.known_issues[:3])}")

        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")

        if self.notes:
            lines.append(f"Notes: {self.notes[:200]}")

        return "\n".join(lines)


class CustomerProfileStore:
    """Persistent storage for customer profiles using SQLite."""

    def __init__(self, db_path: str | Path = "data/customers.db"):
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
                CREATE TABLE IF NOT EXISTS customers (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_interactions INTEGER DEFAULT 0,
                    escalation_count INTEGER DEFAULT 0,
                    resolved_count INTEGER DEFAULT 0,
                    average_sentiment REAL DEFAULT 0.0,
                    last_sentiment REAL DEFAULT 0.0,
                    preferred_language TEXT DEFAULT 'en',
                    account_tier TEXT DEFAULT 'standard',
                    known_issues TEXT DEFAULT '[]',
                    preferences TEXT DEFAULT '{}',
                    tags TEXT DEFAULT '[]',
                    notes TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_customer_email ON customers(email);
                CREATE INDEX IF NOT EXISTS idx_customer_tier ON customers(account_tier);
            """)

    def create_or_get_customer(
        self,
        customer_id: str,
        name: str | None = None,
        email: str | None = None,
    ) -> CustomerProfile:
        """Get existing customer or create new one."""
        existing = self.get_customer(customer_id)
        if existing:
            return existing

        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        profile = CustomerProfile(
            id=customer_id,
            name=name,
            email=email,
            created_at=now,
            updated_at=now,
        )

        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO customers
                   (id, name, email, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (profile.id, profile.name, profile.email, profile.created_at, profile.updated_at),
            )

        return profile

    def get_customer(self, customer_id: str) -> CustomerProfile | None:
        """Retrieve a customer profile."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM customers WHERE id = ?", (customer_id,)
            ).fetchone()

            if not row:
                return None

            return CustomerProfile(
                id=row["id"],
                name=row["name"],
                email=row["email"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                total_interactions=row["total_interactions"],
                escalation_count=row["escalation_count"],
                resolved_count=row["resolved_count"],
                average_sentiment=row["average_sentiment"],
                last_sentiment=row["last_sentiment"],
                preferred_language=row["preferred_language"],
                account_tier=row["account_tier"],
                known_issues=json.loads(row["known_issues"]),
                preferences=json.loads(row["preferences"]),
                tags=json.loads(row["tags"]),
                notes=row["notes"],
            )

    def update_customer(self, profile: CustomerProfile) -> None:
        """Update a customer profile."""
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        with self._get_connection() as conn:
            conn.execute(
                """UPDATE customers SET
                   name = ?, email = ?, updated_at = ?,
                   total_interactions = ?, escalation_count = ?, resolved_count = ?,
                   average_sentiment = ?, last_sentiment = ?,
                   preferred_language = ?, account_tier = ?,
                   known_issues = ?, preferences = ?, tags = ?, notes = ?
                   WHERE id = ?""",
                (
                    profile.name,
                    profile.email,
                    now,
                    profile.total_interactions,
                    profile.escalation_count,
                    profile.resolved_count,
                    profile.average_sentiment,
                    profile.last_sentiment,
                    profile.preferred_language,
                    profile.account_tier,
                    json.dumps(profile.known_issues),
                    json.dumps(profile.preferences),
                    json.dumps(profile.tags),
                    profile.notes,
                    profile.id,
                ),
            )

    def record_interaction(
        self,
        customer_id: str,
        was_escalated: bool = False,
        was_resolved: bool = False,
        sentiment: float | None = None,
    ) -> CustomerProfile:
        """Record an interaction and update statistics."""
        profile = self.create_or_get_customer(customer_id)

        profile.total_interactions += 1
        if was_escalated:
            profile.escalation_count += 1
        if was_resolved:
            profile.resolved_count += 1

        if sentiment is not None:
            # Update running average
            n = profile.total_interactions
            profile.average_sentiment = (
                (profile.average_sentiment * (n - 1) + sentiment) / n
            )
            profile.last_sentiment = sentiment

        self.update_customer(profile)
        return profile

    def add_known_issue(self, customer_id: str, issue: str) -> None:
        """Add a known issue for a customer."""
        profile = self.get_customer(customer_id)
        if profile and issue not in profile.known_issues:
            profile.known_issues.append(issue)
            # Keep only last 10 issues
            profile.known_issues = profile.known_issues[-10:]
            self.update_customer(profile)

    def add_tag(self, customer_id: str, tag: str) -> None:
        """Add a tag to a customer."""
        profile = self.get_customer(customer_id)
        if profile and tag not in profile.tags:
            profile.tags.append(tag)
            self.update_customer(profile)

    def search_by_tag(self, tag: str) -> list[CustomerProfile]:
        """Find customers by tag."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id FROM customers WHERE tags LIKE ?",
                (f'%"{tag}"%',),
            ).fetchall()

        return [
            profile
            for row in rows
            if (profile := self.get_customer(row["id"])) is not None
        ]

    def get_high_escalation_customers(self, threshold: float = 0.3) -> list[CustomerProfile]:
        """Find customers with high escalation rates."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT id FROM customers
                   WHERE total_interactions > 2
                   AND (CAST(escalation_count AS REAL) / total_interactions) > ?""",
                (threshold,),
            ).fetchall()

        return [
            profile
            for row in rows
            if (profile := self.get_customer(row["id"])) is not None
        ]

    def get_negative_sentiment_customers(self, threshold: float = -0.3) -> list[CustomerProfile]:
        """Find customers with negative sentiment."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id FROM customers WHERE last_sentiment < ?",
                (threshold,),
            ).fetchall()

        return [
            profile
            for row in rows
            if (profile := self.get_customer(row["id"])) is not None
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get overall customer statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
            by_tier = dict(
                conn.execute(
                    "SELECT account_tier, COUNT(*) FROM customers GROUP BY account_tier"
                ).fetchall()
            )
            avg_sentiment = conn.execute(
                "SELECT AVG(average_sentiment) FROM customers"
            ).fetchone()[0]

        return {
            "total_customers": total,
            "by_tier": by_tier,
            "average_sentiment": round(avg_sentiment or 0, 3),
        }

    def clear_all(self) -> None:
        """Clear all data (useful for testing)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM customers")
