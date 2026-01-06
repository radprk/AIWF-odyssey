"""Report generation for evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmarks import BenchmarkResult
from .metrics import EvaluationResult


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    name: str
    generated_at: str
    version: str = "1.0.0"

    # Summary statistics
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0

    # Score distributions
    avg_overall_score: float = 0.0
    avg_relevance: float = 0.0
    avg_faithfulness: float = 0.0
    avg_helpfulness: float = 0.0
    avg_tone: float = 0.0
    avg_completeness: float = 0.0
    avg_accuracy: float = 0.0

    # Issue counts
    hallucination_count: int = 0
    pii_exposure_count: int = 0
    wrong_escalation_count: int = 0

    # Performance
    avg_execution_time_ms: float = 0.0
    avg_evaluation_time_ms: float = 0.0

    # Breakdown by category
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Breakdown by difficulty
    by_difficulty: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Individual results
    results: list[dict] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / max(self.total_cases, 1)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "generated_at": self.generated_at,
            "version": self.version,
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.failed_cases,
                "pass_rate": round(self.pass_rate, 3),
            },
            "scores": {
                "avg_overall": self.avg_overall_score,
                "avg_relevance": self.avg_relevance,
                "avg_faithfulness": self.avg_faithfulness,
                "avg_helpfulness": self.avg_helpfulness,
                "avg_tone": self.avg_tone,
                "avg_completeness": self.avg_completeness,
                "avg_accuracy": self.avg_accuracy,
            },
            "issues": {
                "hallucinations": self.hallucination_count,
                "pii_exposures": self.pii_exposure_count,
                "wrong_escalations": self.wrong_escalation_count,
            },
            "performance": {
                "avg_execution_time_ms": self.avg_execution_time_ms,
                "avg_evaluation_time_ms": self.avg_evaluation_time_ms,
            },
            "by_category": self.by_category,
            "by_difficulty": self.by_difficulty,
            "results": self.results,
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Evaluation Report: {self.name}",
            f"Generated: {self.generated_at}",
            "",
            "## Summary",
            f"- **Total Cases**: {self.total_cases}",
            f"- **Passed**: {self.passed_cases} ({self.pass_rate:.1%})",
            f"- **Failed**: {self.failed_cases}",
            "",
            "## Scores",
            f"- Overall: **{self.avg_overall_score:.2f}**/5",
            f"- Relevance: {self.avg_relevance:.2f}",
            f"- Faithfulness: {self.avg_faithfulness:.2f}",
            f"- Helpfulness: {self.avg_helpfulness:.2f}",
            f"- Tone: {self.avg_tone:.2f}",
            f"- Completeness: {self.avg_completeness:.2f}",
            f"- Accuracy: {self.avg_accuracy:.2f}",
            "",
            "## Issues Detected",
            f"- Hallucinations: {self.hallucination_count}",
            f"- PII Exposures: {self.pii_exposure_count}",
            f"- Wrong Escalations: {self.wrong_escalation_count}",
            "",
            "## Performance",
            f"- Avg Execution Time: {self.avg_execution_time_ms:.0f}ms",
            f"- Avg Evaluation Time: {self.avg_evaluation_time_ms:.0f}ms",
            "",
        ]

        if self.by_category:
            lines.append("## Results by Category")
            for cat, stats in self.by_category.items():
                lines.append(f"### {cat.title()}")
                lines.append(f"- Cases: {stats['count']}")
                lines.append(f"- Pass Rate: {stats['pass_rate']:.1%}")
                lines.append(f"- Avg Score: {stats['avg_score']:.2f}")
                lines.append("")

        if self.by_difficulty:
            lines.append("## Results by Difficulty")
            for diff, stats in self.by_difficulty.items():
                lines.append(f"### {diff.title()}")
                lines.append(f"- Cases: {stats['count']}")
                lines.append(f"- Pass Rate: {stats['pass_rate']:.1%}")
                lines.append(f"- Avg Score: {stats['avg_score']:.2f}")
                lines.append("")

        if self.recommendations:
            lines.append("## Recommendations")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Failed cases details
        failed = [r for r in self.results if not r.get("passed", True)]
        if failed:
            lines.append("## Failed Cases")
            for r in failed[:10]:  # Top 10 failures
                lines.append(f"### {r.get('case_name', 'Unknown')}")
                lines.append(f"- Score: {r.get('overall_score', 0):.2f}")
                lines.append(f"- Rationale: {r.get('rationale', 'N/A')}")
                if r.get("suggestions"):
                    lines.append(f"- Suggestions: {', '.join(r['suggestions'])}")
                lines.append("")

        return "\n".join(lines)

    def save(self, path: str | Path, format: str = "json") -> None:
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format == "markdown":
            with open(path, "w") as f:
                f.write(self.to_markdown())


def generate_evaluation_report(
    results: list[BenchmarkResult],
    name: str = "Evaluation Report",
) -> EvaluationReport:
    """Generate a comprehensive report from benchmark results."""
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    report = EvaluationReport(name=name, generated_at=now)
    report.total_cases = len(results)
    report.passed_cases = sum(1 for r in results if r.passed)
    report.failed_cases = report.total_cases - report.passed_cases

    if not results:
        return report

    # Calculate averages
    report.avg_overall_score = sum(r.evaluation.overall_score for r in results) / len(results)
    report.avg_relevance = sum(r.evaluation.relevance for r in results) / len(results)
    report.avg_faithfulness = sum(r.evaluation.faithfulness for r in results) / len(results)
    report.avg_helpfulness = sum(r.evaluation.helpfulness for r in results) / len(results)
    report.avg_tone = sum(r.evaluation.tone_appropriateness for r in results) / len(results)
    report.avg_completeness = sum(r.evaluation.completeness for r in results) / len(results)
    report.avg_accuracy = sum(r.evaluation.accuracy for r in results) / len(results)

    # Round scores
    for attr in ["avg_overall_score", "avg_relevance", "avg_faithfulness",
                 "avg_helpfulness", "avg_tone", "avg_completeness", "avg_accuracy"]:
        setattr(report, attr, round(getattr(report, attr), 2))

    # Count issues
    report.hallucination_count = sum(1 for r in results if r.evaluation.contains_hallucination)
    report.pii_exposure_count = sum(1 for r in results if r.evaluation.pii_exposed)
    report.wrong_escalation_count = sum(1 for r in results if not r.escalation_correct)

    # Performance
    report.avg_execution_time_ms = round(
        sum(r.execution_time_ms for r in results) / len(results), 1
    )
    report.avg_evaluation_time_ms = round(
        sum(r.evaluation.processing_time_ms for r in results) / len(results), 1
    )

    # Group by category
    categories: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        cat = r.case.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, cat_results in categories.items():
        passed = sum(1 for r in cat_results if r.passed)
        report.by_category[cat] = {
            "count": len(cat_results),
            "passed": passed,
            "pass_rate": passed / len(cat_results),
            "avg_score": round(
                sum(r.evaluation.overall_score for r in cat_results) / len(cat_results), 2
            ),
        }

    # Group by difficulty
    difficulties: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        diff = r.case.difficulty
        if diff not in difficulties:
            difficulties[diff] = []
        difficulties[diff].append(r)

    for diff, diff_results in difficulties.items():
        passed = sum(1 for r in diff_results if r.passed)
        report.by_difficulty[diff] = {
            "count": len(diff_results),
            "passed": passed,
            "pass_rate": passed / len(diff_results),
            "avg_score": round(
                sum(r.evaluation.overall_score for r in diff_results) / len(diff_results), 2
            ),
        }

    # Individual results
    for r in results:
        report.results.append({
            "case_id": r.case.id,
            "case_name": r.case.name,
            "category": r.case.category,
            "difficulty": r.case.difficulty,
            "passed": r.passed,
            "overall_score": r.evaluation.overall_score,
            "rationale": r.evaluation.rationale,
            "suggestions": r.evaluation.suggestions,
            "execution_time_ms": r.execution_time_ms,
        })

    # Generate recommendations
    report.recommendations = _generate_recommendations(report)

    return report


def _generate_recommendations(report: EvaluationReport) -> list[str]:
    """Generate actionable recommendations based on report."""
    recommendations = []

    if report.hallucination_count > 0:
        recommendations.append(
            f"Address hallucinations: {report.hallucination_count} cases had fabricated information. "
            "Consider improving RAG context retrieval or adding fact-checking."
        )

    if report.pii_exposure_count > 0:
        recommendations.append(
            f"Fix PII exposure: {report.pii_exposure_count} cases leaked personal information. "
            "Add PII detection and redaction to response pipeline."
        )

    if report.wrong_escalation_count > 2:
        recommendations.append(
            f"Improve escalation logic: {report.wrong_escalation_count} cases had incorrect escalation decisions. "
            "Review escalation triggers and thresholds."
        )

    if report.avg_tone < 3.5:
        recommendations.append(
            f"Improve response tone: Average tone score is {report.avg_tone:.1f}/5. "
            "Update agent prompts to emphasize empathy and professionalism."
        )

    if report.avg_completeness < 3.5:
        recommendations.append(
            f"Improve completeness: Average completeness score is {report.avg_completeness:.1f}/5. "
            "Ensure agents fully address all aspects of customer queries."
        )

    # Category-specific recommendations
    for cat, stats in report.by_category.items():
        if stats["pass_rate"] < 0.6:
            recommendations.append(
                f"Focus on {cat} category: Only {stats['pass_rate']:.0%} pass rate. "
                f"Review {cat} specialist training and knowledge base."
            )

    # Difficulty-specific recommendations
    hard_stats = report.by_difficulty.get("hard", {})
    if hard_stats and hard_stats.get("pass_rate", 1) < 0.5:
        recommendations.append(
            "Improve handling of complex cases: Hard difficulty pass rate is low. "
            "Consider adding L3 expert involvement or enhanced context for complex queries."
        )

    if not recommendations:
        recommendations.append("All metrics look healthy. Continue monitoring for regressions.")

    return recommendations


def compare_reports(
    baseline: EvaluationReport,
    current: EvaluationReport,
) -> dict[str, Any]:
    """Compare two evaluation reports for regression detection."""
    return {
        "baseline_name": baseline.name,
        "current_name": current.name,
        "baseline_date": baseline.generated_at,
        "current_date": current.generated_at,
        "pass_rate_change": round(current.pass_rate - baseline.pass_rate, 3),
        "score_change": round(current.avg_overall_score - baseline.avg_overall_score, 2),
        "hallucination_change": current.hallucination_count - baseline.hallucination_count,
        "pii_change": current.pii_exposure_count - baseline.pii_exposure_count,
        "is_regression": (
            current.pass_rate < baseline.pass_rate - 0.05 or
            current.avg_overall_score < baseline.avg_overall_score - 0.2 or
            current.hallucination_count > baseline.hallucination_count
        ),
        "category_changes": {
            cat: {
                "pass_rate_change": round(
                    current.by_category.get(cat, {}).get("pass_rate", 0) -
                    stats.get("pass_rate", 0), 3
                ),
                "score_change": round(
                    current.by_category.get(cat, {}).get("avg_score", 0) -
                    stats.get("avg_score", 0), 2
                ),
            }
            for cat, stats in baseline.by_category.items()
        },
    }
