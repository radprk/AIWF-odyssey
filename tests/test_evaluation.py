"""Tests for evaluation system."""

from __future__ import annotations

import pytest


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_overall_score_calculation(self):
        """Test weighted overall score calculation."""
        from evaluation.metrics import EvaluationResult

        result = EvaluationResult(
            relevance=5.0,
            faithfulness=5.0,
            helpfulness=5.0,
            tone_appropriateness=5.0,
            completeness=5.0,
            accuracy=5.0,
            clarity=5.0,
            actionability=5.0,
        )

        assert result.overall_score == 5.0

    def test_overall_score_with_penalties(self):
        """Test penalties reduce overall score."""
        from evaluation.metrics import EvaluationResult

        base_result = EvaluationResult(
            relevance=5.0, faithfulness=5.0, helpfulness=5.0,
            tone_appropriateness=5.0, completeness=5.0, accuracy=5.0,
            clarity=5.0, actionability=5.0,
        )

        hallucination_result = EvaluationResult(
            relevance=5.0, faithfulness=5.0, helpfulness=5.0,
            tone_appropriateness=5.0, completeness=5.0, accuracy=5.0,
            clarity=5.0, actionability=5.0,
            contains_hallucination=True,
        )

        assert hallucination_result.overall_score < base_result.overall_score

    def test_pii_exposure_severe_penalty(self):
        """Test PII exposure has severe penalty."""
        from evaluation.metrics import EvaluationResult

        result = EvaluationResult(
            relevance=5.0, faithfulness=5.0, helpfulness=5.0,
            tone_appropriateness=5.0, completeness=5.0, accuracy=5.0,
            clarity=5.0, actionability=5.0,
            pii_exposed=True,
        )

        # 50% penalty
        assert result.overall_score == 2.5

    def test_passed_threshold(self):
        """Test passed property threshold."""
        from evaluation.metrics import EvaluationResult

        passing = EvaluationResult(
            relevance=4.0, faithfulness=4.0, helpfulness=4.0,
            tone_appropriateness=4.0, completeness=4.0, accuracy=4.0,
            clarity=4.0, actionability=4.0,
        )

        failing = EvaluationResult(
            relevance=2.0, faithfulness=2.0, helpfulness=2.0,
            tone_appropriateness=2.0, completeness=2.0, accuracy=2.0,
            clarity=2.0, actionability=2.0,
        )

        assert passing.passed is True
        assert failing.passed is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from evaluation.metrics import EvaluationResult

        result = EvaluationResult(relevance=4.0, helpfulness=3.5)
        data = result.to_dict()

        assert data["relevance"] == 4.0
        assert data["helpfulness"] == 3.5
        assert "overall_score" in data
        assert "passed" in data


class TestLightweightChecks:
    """Tests for rule-based checks (no LLM required)."""

    def test_check_pii_ssn(self):
        """Test SSN detection."""
        from evaluation.metrics import check_pii_exposure

        text_with_ssn = "Your SSN is 123-45-6789"
        text_without = "Your account number is 12345"

        assert "ssn" in check_pii_exposure(text_with_ssn)
        assert "ssn" not in check_pii_exposure(text_without)

    def test_check_pii_credit_card(self):
        """Test credit card detection."""
        from evaluation.metrics import check_pii_exposure

        text = "Card number: 4111-1111-1111-1111"
        assert "credit_card" in check_pii_exposure(text)

    def test_check_pii_email(self):
        """Test email detection."""
        from evaluation.metrics import check_pii_exposure

        text = "Contact john.doe@example.com for help"
        assert "email" in check_pii_exposure(text)

    def test_check_pii_phone(self):
        """Test phone number detection."""
        from evaluation.metrics import check_pii_exposure

        text = "Call us at 555-123-4567"
        assert "phone" in check_pii_exposure(text)

    def test_check_response_length(self):
        """Test response length checking."""
        from evaluation.metrics import check_response_length

        short = "Yes"
        appropriate = " ".join(["word"] * 50)
        long = " ".join(["word"] * 600)

        assert check_response_length(short)["too_short"] is True
        assert check_response_length(appropriate)["appropriate"] is True
        assert check_response_length(long)["too_long"] is True

    def test_check_sentiment(self):
        """Test simple sentiment detection."""
        from evaluation.metrics import check_sentiment

        negative = "I am so frustrated and angry with this terrible service!"
        positive = "Thank you so much, this was excellent and very helpful!"
        neutral = "I would like to know about my account."

        assert check_sentiment(negative) == "negative"
        assert check_sentiment(positive) == "positive"
        assert check_sentiment(neutral) == "neutral"


class TestBenchmarkCase:
    """Tests for BenchmarkCase."""

    def test_benchmark_case_creation(self):
        """Test creating a benchmark case."""
        from evaluation.benchmarks import BenchmarkCase

        case = BenchmarkCase(
            id="test-001",
            name="Test Case",
            category="billing",
            difficulty="easy",
            query="Test query",
            expected_outcome="Expected response",
            should_escalate=False,
            min_overall=3.5,
        )

        assert case.id == "test-001"
        assert case.category == "billing"
        assert case.min_overall == 3.5

    def test_benchmark_case_to_dict(self):
        """Test serialization."""
        from evaluation.benchmarks import BenchmarkCase

        case = BenchmarkCase(
            id="test-002",
            name="Test",
            category="account",
            difficulty="medium",
            query="Query",
            tags=["test", "unit"],
        )

        data = case.to_dict()
        assert data["id"] == "test-002"
        assert "test" in data["tags"]


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_suite_creation(self):
        """Test creating a benchmark suite."""
        from evaluation.benchmarks import BenchmarkSuite, BenchmarkCase

        suite = BenchmarkSuite(
            name="Test Suite",
            description="For testing",
        )

        suite.add_case(BenchmarkCase(
            id="case-001",
            name="Case 1",
            category="billing",
            difficulty="easy",
            query="Query 1",
        ))

        assert len(suite.cases) == 1

    def test_filter_by_category(self):
        """Test filtering by category."""
        from evaluation.benchmarks import BenchmarkSuite, BenchmarkCase

        suite = BenchmarkSuite(name="Test", description="Test")
        suite.add_case(BenchmarkCase("1", "C1", "billing", "easy", "Q1"))
        suite.add_case(BenchmarkCase("2", "C2", "account", "easy", "Q2"))
        suite.add_case(BenchmarkCase("3", "C3", "billing", "medium", "Q3"))

        billing = suite.filter_by_category("billing")
        assert len(billing) == 2

    def test_filter_by_difficulty(self):
        """Test filtering by difficulty."""
        from evaluation.benchmarks import BenchmarkSuite, BenchmarkCase

        suite = BenchmarkSuite(name="Test", description="Test")
        suite.add_case(BenchmarkCase("1", "C1", "billing", "easy", "Q1"))
        suite.add_case(BenchmarkCase("2", "C2", "billing", "hard", "Q2"))

        easy = suite.filter_by_difficulty("easy")
        assert len(easy) == 1

    def test_save_and_load(self, temp_dir):
        """Test saving and loading suite."""
        from evaluation.benchmarks import BenchmarkSuite, BenchmarkCase

        suite = BenchmarkSuite(name="Save Test", description="Test saving")
        suite.add_case(BenchmarkCase("1", "Case", "billing", "easy", "Query"))

        path = temp_dir / "suite.json"
        suite.save(path)

        loaded = BenchmarkSuite.load(path)
        assert loaded.name == "Save Test"
        assert len(loaded.cases) == 1

    def test_load_default_benchmarks(self):
        """Test loading default benchmark suite."""
        from evaluation.benchmarks import load_default_benchmarks

        suite = load_default_benchmarks()

        assert suite.name == "Customer Support Benchmarks"
        assert len(suite.cases) > 0

        # Check categories are covered
        categories = {c.category for c in suite.cases}
        assert "billing" in categories
        assert "account" in categories
        assert "transaction" in categories


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_result_passed_criteria(self):
        """Test passed property checks all criteria."""
        from evaluation.benchmarks import BenchmarkCase, BenchmarkResult
        from evaluation.metrics import EvaluationResult

        case = BenchmarkCase(
            id="test",
            name="Test",
            category="billing",
            difficulty="easy",
            query="Query",
            min_relevance=3.0,
            min_helpfulness=3.0,
            min_overall=3.5,
        )

        passing_eval = EvaluationResult(
            relevance=4.0, faithfulness=4.0, helpfulness=4.0,
            tone_appropriateness=4.0, completeness=4.0, accuracy=4.0,
            clarity=4.0, actionability=4.0,
        )

        result = BenchmarkResult(
            case=case,
            response="Good response",
            evaluation=passing_eval,
            actual_specialists=["billing"],
            was_escalated=False,
            execution_time_ms=100,
            timestamp="2024-01-01T00:00:00Z",
        )

        assert result.passed is True

    def test_result_fails_on_hallucination(self):
        """Test hallucination causes failure."""
        from evaluation.benchmarks import BenchmarkCase, BenchmarkResult
        from evaluation.metrics import EvaluationResult

        case = BenchmarkCase("test", "Test", "billing", "easy", "Query")

        eval_with_hallucination = EvaluationResult(
            relevance=4.0, faithfulness=4.0, helpfulness=4.0,
            tone_appropriateness=4.0, completeness=4.0, accuracy=4.0,
            clarity=4.0, actionability=4.0,
            contains_hallucination=True,
        )

        result = BenchmarkResult(
            case=case,
            response="Response",
            evaluation=eval_with_hallucination,
            actual_specialists=[],
            was_escalated=False,
            execution_time_ms=100,
            timestamp="2024-01-01T00:00:00Z",
        )

        assert result.passed is False

    def test_escalation_correctness(self):
        """Test escalation correctness check."""
        from evaluation.benchmarks import BenchmarkCase, BenchmarkResult
        from evaluation.metrics import EvaluationResult

        case_should_escalate = BenchmarkCase(
            "test", "Test", "billing", "hard", "Query",
            should_escalate=True,
        )

        eval_result = EvaluationResult(relevance=4.0, helpfulness=4.0)

        # Correct: should escalate and did escalate
        result_correct = BenchmarkResult(
            case=case_should_escalate,
            response="Response",
            evaluation=eval_result,
            actual_specialists=[],
            was_escalated=True,
            execution_time_ms=100,
            timestamp="2024-01-01T00:00:00Z",
        )

        # Incorrect: should escalate but didn't
        result_incorrect = BenchmarkResult(
            case=case_should_escalate,
            response="Response",
            evaluation=eval_result,
            actual_specialists=[],
            was_escalated=False,
            execution_time_ms=100,
            timestamp="2024-01-01T00:00:00Z",
        )

        assert result_correct.escalation_correct is True
        assert result_incorrect.escalation_correct is False


class TestEvaluationReport:
    """Tests for EvaluationReport."""

    def test_generate_report(self):
        """Test generating report from benchmark results."""
        from evaluation.benchmarks import BenchmarkCase, BenchmarkResult
        from evaluation.metrics import EvaluationResult
        from evaluation.reports import generate_evaluation_report

        case = BenchmarkCase("test", "Test", "billing", "easy", "Query")
        eval_result = EvaluationResult(
            relevance=4.0, faithfulness=4.0, helpfulness=4.0,
            tone_appropriateness=4.0, completeness=4.0, accuracy=4.0,
            clarity=4.0, actionability=4.0,
            processing_time_ms=50,
        )

        results = [
            BenchmarkResult(
                case=case,
                response="Response",
                evaluation=eval_result,
                actual_specialists=["billing"],
                was_escalated=False,
                execution_time_ms=100,
                timestamp="2024-01-01T00:00:00Z",
            )
        ]

        report = generate_evaluation_report(results, "Test Report")

        assert report.name == "Test Report"
        assert report.total_cases == 1
        assert report.passed_cases == 1
        assert report.pass_rate == 1.0

    def test_report_with_failures(self):
        """Test report with some failures."""
        from evaluation.benchmarks import BenchmarkCase, BenchmarkResult
        from evaluation.metrics import EvaluationResult
        from evaluation.reports import generate_evaluation_report

        good_eval = EvaluationResult(
            relevance=4.0, faithfulness=4.0, helpfulness=4.0,
            tone_appropriateness=4.0, completeness=4.0, accuracy=4.0,
            clarity=4.0, actionability=4.0,
        )

        bad_eval = EvaluationResult(
            relevance=2.0, faithfulness=2.0, helpfulness=2.0,
            tone_appropriateness=2.0, completeness=2.0, accuracy=2.0,
            clarity=2.0, actionability=2.0,
            contains_hallucination=True,
        )

        case = BenchmarkCase("test", "Test", "billing", "easy", "Query")

        results = [
            BenchmarkResult(case, "Good", good_eval, [], False, 100, "2024-01-01T00:00:00Z"),
            BenchmarkResult(case, "Bad", bad_eval, [], False, 100, "2024-01-01T00:00:00Z"),
        ]

        report = generate_evaluation_report(results)

        assert report.total_cases == 2
        assert report.passed_cases == 1
        assert report.failed_cases == 1
        assert report.hallucination_count == 1

    def test_report_recommendations(self):
        """Test that recommendations are generated."""
        from evaluation.benchmarks import BenchmarkCase, BenchmarkResult
        from evaluation.metrics import EvaluationResult
        from evaluation.reports import generate_evaluation_report

        # Create result with issues
        bad_eval = EvaluationResult(
            relevance=2.0, faithfulness=2.0, helpfulness=2.0,
            tone_appropriateness=2.0, completeness=2.0, accuracy=2.0,
            clarity=2.0, actionability=2.0,
            contains_hallucination=True,
            pii_exposed=True,
        )

        case = BenchmarkCase("test", "Test", "billing", "easy", "Query")
        results = [
            BenchmarkResult(case, "Bad", bad_eval, [], False, 100, "2024-01-01T00:00:00Z"),
        ]

        report = generate_evaluation_report(results)

        assert len(report.recommendations) > 0
        # Should recommend fixing hallucinations and PII
        rec_text = " ".join(report.recommendations).lower()
        assert "hallucination" in rec_text or "pii" in rec_text

    def test_report_to_markdown(self):
        """Test markdown report generation."""
        from evaluation.benchmarks import BenchmarkCase, BenchmarkResult
        from evaluation.metrics import EvaluationResult
        from evaluation.reports import generate_evaluation_report

        eval_result = EvaluationResult(relevance=4.0, helpfulness=4.0)
        case = BenchmarkCase("test", "Test", "billing", "easy", "Query")
        results = [
            BenchmarkResult(case, "Response", eval_result, [], False, 100, "2024-01-01T00:00:00Z"),
        ]

        report = generate_evaluation_report(results, "MD Test")
        markdown = report.to_markdown()

        assert "# Evaluation Report: MD Test" in markdown
        assert "## Summary" in markdown
        assert "## Scores" in markdown

    def test_report_save_json(self, temp_dir):
        """Test saving report as JSON."""
        from evaluation.reports import EvaluationReport

        report = EvaluationReport(
            name="Save Test",
            generated_at="2024-01-01T00:00:00Z",
            total_cases=5,
            passed_cases=4,
        )

        path = temp_dir / "report.json"
        report.save(path, format="json")

        assert path.exists()

    def test_compare_reports(self):
        """Test comparing two reports."""
        from evaluation.reports import EvaluationReport, compare_reports

        baseline = EvaluationReport(
            name="Baseline",
            generated_at="2024-01-01T00:00:00Z",
            total_cases=10,
            passed_cases=8,
            avg_overall_score=4.0,
            hallucination_count=1,
        )

        current = EvaluationReport(
            name="Current",
            generated_at="2024-01-02T00:00:00Z",
            total_cases=10,
            passed_cases=6,
            avg_overall_score=3.5,
            hallucination_count=3,
        )

        comparison = compare_reports(baseline, current)

        assert comparison["is_regression"] is True
        assert comparison["pass_rate_change"] < 0
        assert comparison["hallucination_change"] == 2
