"""Benchmark system for systematic evaluation of agent responses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .metrics import EvaluationResult, ResponseEvaluator


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    id: str
    name: str
    category: str  # "billing", "account", "transaction", etc.
    difficulty: str  # "easy", "medium", "hard"

    query: str
    context: str = ""
    expected_outcome: str = ""

    # Expected behavior
    should_escalate: bool = False
    expected_specialists: list[str] = field(default_factory=list)

    # Quality thresholds
    min_relevance: float = 3.0
    min_helpfulness: float = 3.0
    min_overall: float = 3.5

    # Tags for filtering
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "difficulty": self.difficulty,
            "query": self.query,
            "context": self.context,
            "expected_outcome": self.expected_outcome,
            "should_escalate": self.should_escalate,
            "expected_specialists": self.expected_specialists,
            "min_relevance": self.min_relevance,
            "min_helpfulness": self.min_helpfulness,
            "min_overall": self.min_overall,
            "tags": self.tags,
        }


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark case."""
    case: BenchmarkCase
    response: str
    evaluation: EvaluationResult
    actual_specialists: list[str]
    was_escalated: bool
    execution_time_ms: float
    timestamp: str

    @property
    def passed(self) -> bool:
        """Check if benchmark passed all criteria."""
        return (
            self.evaluation.relevance >= self.case.min_relevance
            and self.evaluation.helpfulness >= self.case.min_helpfulness
            and self.evaluation.overall_score >= self.case.min_overall
            and not self.evaluation.contains_hallucination
            and not self.evaluation.pii_exposed
        )

    @property
    def escalation_correct(self) -> bool:
        """Check if escalation behavior was correct."""
        return self.was_escalated == self.case.should_escalate

    def to_dict(self) -> dict:
        return {
            "case_id": self.case.id,
            "case_name": self.case.name,
            "passed": self.passed,
            "escalation_correct": self.escalation_correct,
            "response": self.response,
            "evaluation": self.evaluation.to_dict(),
            "actual_specialists": self.actual_specialists,
            "was_escalated": self.was_escalated,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark cases."""
    name: str
    description: str
    cases: list[BenchmarkCase] = field(default_factory=list)
    version: str = "1.0.0"

    def add_case(self, case: BenchmarkCase) -> None:
        self.cases.append(case)

    def filter_by_category(self, category: str) -> list[BenchmarkCase]:
        return [c for c in self.cases if c.category == category]

    def filter_by_difficulty(self, difficulty: str) -> list[BenchmarkCase]:
        return [c for c in self.cases if c.difficulty == difficulty]

    def filter_by_tag(self, tag: str) -> list[BenchmarkCase]:
        return [c for c in self.cases if tag in c.tags]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "cases": [c.to_dict() for c in self.cases],
        }

    def save(self, path: str | Path) -> None:
        """Save benchmark suite to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkSuite":
        """Load benchmark suite from JSON file."""
        with open(path) as f:
            data = json.load(f)

        suite = cls(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
        )

        for case_data in data.get("cases", []):
            suite.add_case(BenchmarkCase(
                id=case_data["id"],
                name=case_data["name"],
                category=case_data["category"],
                difficulty=case_data["difficulty"],
                query=case_data["query"],
                context=case_data.get("context", ""),
                expected_outcome=case_data.get("expected_outcome", ""),
                should_escalate=case_data.get("should_escalate", False),
                expected_specialists=case_data.get("expected_specialists", []),
                min_relevance=case_data.get("min_relevance", 3.0),
                min_helpfulness=case_data.get("min_helpfulness", 3.0),
                min_overall=case_data.get("min_overall", 3.5),
                tags=case_data.get("tags", []),
            ))

        return suite


def load_default_benchmarks() -> BenchmarkSuite:
    """Load the default benchmark suite for customer support."""
    suite = BenchmarkSuite(
        name="Customer Support Benchmarks",
        description="Standard benchmark cases for evaluating customer support agents",
    )

    # Billing category
    suite.add_case(BenchmarkCase(
        id="billing-001",
        name="Simple fee inquiry",
        category="billing",
        difficulty="easy",
        query="Why was I charged a $15 fee on my account?",
        expected_outcome="Explain common fee types and offer to investigate specific charge",
        tags=["fees", "inquiry"],
    ))

    suite.add_case(BenchmarkCase(
        id="billing-002",
        name="Fee dispute",
        category="billing",
        difficulty="medium",
        query="I want to dispute the $50 overdraft fee. This is the third time this month and I feel it's unfair.",
        expected_outcome="Acknowledge frustration, explain policy, offer goodwill credit or escalate",
        should_escalate=True,
        tags=["fees", "dispute", "escalation"],
    ))

    suite.add_case(BenchmarkCase(
        id="billing-003",
        name="Complex billing dispute",
        category="billing",
        difficulty="hard",
        query="I've been charged incorrectly for 3 months. I have statements showing the discrepancy. I want a full refund and compensation for my time.",
        expected_outcome="Take detailed information, escalate to supervisor, set clear expectations",
        should_escalate=True,
        min_overall=4.0,
        tags=["fees", "dispute", "compensation", "escalation"],
    ))

    # Account category
    suite.add_case(BenchmarkCase(
        id="account-001",
        name="Password reset",
        category="account",
        difficulty="easy",
        query="I forgot my password and need to reset it.",
        expected_outcome="Provide clear password reset instructions",
        tags=["password", "access"],
    ))

    suite.add_case(BenchmarkCase(
        id="account-002",
        name="Account locked",
        category="account",
        difficulty="medium",
        query="My account has been locked and I can't access my funds. This is urgent!",
        expected_outcome="Verify identity, explain unlock process, prioritize due to urgency",
        tags=["access", "urgent", "security"],
    ))

    suite.add_case(BenchmarkCase(
        id="account-003",
        name="Fraudulent activity",
        category="account",
        difficulty="hard",
        query="I see transactions I didn't make. Someone has hacked my account!",
        expected_outcome="Immediate security measures, escalate to fraud team, document details",
        should_escalate=True,
        min_overall=4.0,
        tags=["fraud", "security", "escalation", "urgent"],
    ))

    # Transaction category
    suite.add_case(BenchmarkCase(
        id="transaction-001",
        name="Transfer status",
        category="transaction",
        difficulty="easy",
        query="When will my transfer to my savings account be completed?",
        expected_outcome="Explain typical transfer times and how to check status",
        tags=["transfer", "inquiry"],
    ))

    suite.add_case(BenchmarkCase(
        id="transaction-002",
        name="Failed transaction",
        category="transaction",
        difficulty="medium",
        query="My payment to a vendor failed but the money left my account. Where is it?",
        expected_outcome="Explain pending/failed transaction process, timeline for resolution",
        tags=["payment", "failed", "investigation"],
    ))

    suite.add_case(BenchmarkCase(
        id="transaction-003",
        name="Large wire transfer",
        category="transaction",
        difficulty="hard",
        query="I need to wire $50,000 internationally today for a business deal. What's the process?",
        expected_outcome="Explain wire process, verification requirements, fees, and timing",
        min_overall=4.0,
        tags=["wire", "international", "large-amount"],
    ))

    # General category
    suite.add_case(BenchmarkCase(
        id="general-001",
        name="Product information",
        category="general",
        difficulty="easy",
        query="What types of savings accounts do you offer?",
        expected_outcome="List savings account options with key features",
        tags=["products", "savings", "inquiry"],
    ))

    suite.add_case(BenchmarkCase(
        id="general-002",
        name="Complaint handling",
        category="complaint",
        difficulty="medium",
        query="I've had terrible service today. I've been on hold for an hour and no one can help me!",
        expected_outcome="Apologize sincerely, acknowledge frustration, offer immediate assistance",
        tags=["complaint", "service", "emotional"],
    ))

    suite.add_case(BenchmarkCase(
        id="general-003",
        name="Regulatory inquiry",
        category="general",
        difficulty="hard",
        query="I need documentation proving my account meets regulatory requirements for my business audit.",
        expected_outcome="Explain available documentation, compliance process, escalate if needed",
        should_escalate=True,
        tags=["compliance", "documentation", "business"],
    ))

    return suite


def run_benchmark_suite(
    suite: BenchmarkSuite,
    query_handler: Callable[[str], tuple[str, list[str], bool]],
    evaluator: ResponseEvaluator | None = None,
    categories: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> list[BenchmarkResult]:
    """
    Run a benchmark suite against a query handler.

    Args:
        suite: The benchmark suite to run
        query_handler: Function that takes query and returns (response, specialists, was_escalated)
        evaluator: Optional custom evaluator
        categories: Filter to specific categories
        difficulties: Filter to specific difficulties

    Returns:
        List of benchmark results
    """
    import time

    evaluator = evaluator or ResponseEvaluator()
    results = []

    cases = suite.cases
    if categories:
        cases = [c for c in cases if c.category in categories]
    if difficulties:
        cases = [c for c in cases if c.difficulty in difficulties]

    for case in cases:
        start = time.time()

        # Run query through handler
        response, specialists, was_escalated = query_handler(case.query)

        execution_time = (time.time() - start) * 1000

        # Evaluate response
        evaluation = evaluator.evaluate(
            query=case.query,
            response=response,
            context=case.context,
            expected_outcome=case.expected_outcome,
        )

        result = BenchmarkResult(
            case=case,
            response=response,
            evaluation=evaluation,
            actual_specialists=specialists,
            was_escalated=was_escalated,
            execution_time_ms=execution_time,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )

        results.append(result)

    return results


def compare_benchmark_runs(
    baseline: list[BenchmarkResult],
    current: list[BenchmarkResult],
) -> dict[str, Any]:
    """Compare two benchmark runs for regression detection."""
    baseline_by_id = {r.case.id: r for r in baseline}
    current_by_id = {r.case.id: r for r in current}

    common_ids = set(baseline_by_id.keys()) & set(current_by_id.keys())

    regressions = []
    improvements = []

    for case_id in common_ids:
        b = baseline_by_id[case_id]
        c = current_by_id[case_id]

        score_diff = c.evaluation.overall_score - b.evaluation.overall_score

        if score_diff < -0.5:  # Significant regression
            regressions.append({
                "case_id": case_id,
                "case_name": b.case.name,
                "baseline_score": b.evaluation.overall_score,
                "current_score": c.evaluation.overall_score,
                "diff": score_diff,
            })
        elif score_diff > 0.5:  # Significant improvement
            improvements.append({
                "case_id": case_id,
                "case_name": b.case.name,
                "baseline_score": b.evaluation.overall_score,
                "current_score": c.evaluation.overall_score,
                "diff": score_diff,
            })

    baseline_pass_rate = sum(1 for r in baseline if r.passed) / len(baseline) if baseline else 0
    current_pass_rate = sum(1 for r in current if r.passed) / len(current) if current else 0

    return {
        "baseline_cases": len(baseline),
        "current_cases": len(current),
        "common_cases": len(common_ids),
        "baseline_pass_rate": round(baseline_pass_rate, 3),
        "current_pass_rate": round(current_pass_rate, 3),
        "pass_rate_change": round(current_pass_rate - baseline_pass_rate, 3),
        "regressions": regressions,
        "improvements": improvements,
        "has_regressions": len(regressions) > 0,
    }
