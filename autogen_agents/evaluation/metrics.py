"""Extended evaluation metrics for response quality assessment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from autogen import AssistantAgent, UserProxyAgent


@dataclass
class EvaluationResult:
    """Comprehensive evaluation result with multiple metrics."""

    # Core metrics (1-5 scale)
    relevance: float = 0.0
    faithfulness: float = 0.0
    helpfulness: float = 0.0

    # Extended metrics (1-5 scale)
    tone_appropriateness: float = 0.0  # Professional, empathetic
    completeness: float = 0.0  # Did it fully address the query?
    accuracy: float = 0.0  # Factual correctness
    clarity: float = 0.0  # Easy to understand
    actionability: float = 0.0  # Clear next steps provided

    # Binary assessments
    contains_hallucination: bool = False
    escalation_appropriate: bool = True
    pii_exposed: bool = False

    # Metadata
    rationale: str = ""
    suggestions: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            "relevance": 0.20,
            "faithfulness": 0.15,
            "helpfulness": 0.20,
            "tone_appropriateness": 0.10,
            "completeness": 0.15,
            "accuracy": 0.10,
            "clarity": 0.05,
            "actionability": 0.05,
        }
        score = sum(
            getattr(self, metric) * weight for metric, weight in weights.items()
        )
        # Penalties
        if self.contains_hallucination:
            score *= 0.7
        if self.pii_exposed:
            score *= 0.5
        if not self.escalation_appropriate:
            score *= 0.9
        return round(score, 2)

    @property
    def passed(self) -> bool:
        """Check if response meets quality threshold."""
        return (
            self.overall_score >= 3.5
            and not self.contains_hallucination
            and not self.pii_exposed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevance": self.relevance,
            "faithfulness": self.faithfulness,
            "helpfulness": self.helpfulness,
            "tone_appropriateness": self.tone_appropriateness,
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "clarity": self.clarity,
            "actionability": self.actionability,
            "contains_hallucination": self.contains_hallucination,
            "escalation_appropriate": self.escalation_appropriate,
            "pii_exposed": self.pii_exposed,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "rationale": self.rationale,
            "suggestions": self.suggestions,
            "processing_time_ms": self.processing_time_ms,
        }


EVALUATOR_SYSTEM_MESSAGE = """You are an expert evaluator for customer support responses.

Evaluate the response on these metrics (1-5 scale, where 5 is best):

1. **Relevance**: Does the response address the customer's actual question?
2. **Faithfulness**: Does it use provided context accurately without hallucinating?
3. **Helpfulness**: Does it genuinely help solve the customer's problem?
4. **Tone Appropriateness**: Is the tone professional, empathetic, and appropriate?
5. **Completeness**: Does it fully address all aspects of the query?
6. **Accuracy**: Are all stated facts correct based on the provided context?
7. **Clarity**: Is the response easy to understand?
8. **Actionability**: Are there clear next steps for the customer?

Also assess:
- **contains_hallucination**: true if the response includes made-up information not in the context
- **escalation_appropriate**: true if escalation decisions (or lack thereof) were appropriate
- **pii_exposed**: true if the response inappropriately exposes personal information

Return ONLY valid JSON with this exact structure:
{
    "relevance": <1-5>,
    "faithfulness": <1-5>,
    "helpfulness": <1-5>,
    "tone_appropriateness": <1-5>,
    "completeness": <1-5>,
    "accuracy": <1-5>,
    "clarity": <1-5>,
    "actionability": <1-5>,
    "contains_hallucination": <true/false>,
    "escalation_appropriate": <true/false>,
    "pii_exposed": <true/false>,
    "rationale": "<brief explanation>",
    "suggestions": ["<improvement 1>", "<improvement 2>"]
}
"""


class ResponseEvaluator:
    """Evaluates response quality using an LLM judge."""

    def __init__(self, llm_config: dict | None = None):
        self.llm_config = llm_config or self._default_config()

    def _default_config(self) -> dict:
        return {
            "config_list": [
                {
                    "model": "mistral",
                    "base_url": "http://localhost:11434/v1",
                    "api_key": "ollama",
                    "price": [0, 0],
                }
            ],
            "temperature": 0.1,
        }

    def evaluate(
        self,
        query: str,
        response: str,
        context: str = "",
        expected_outcome: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a response against query and context."""
        import time

        start = time.time()

        evaluator = AssistantAgent(
            "ResponseEvaluator",
            system_message=EVALUATOR_SYSTEM_MESSAGE,
            llm_config=self.llm_config,
        )

        proxy = UserProxyAgent(
            name="EvalProxy",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False},
        )

        prompt_parts = [
            "## Customer Query",
            query,
            "",
            "## Response to Evaluate",
            response,
        ]

        if context:
            prompt_parts.extend(["", "## Available Context/Knowledge Base", context])

        if expected_outcome:
            prompt_parts.extend(["", "## Expected Outcome", expected_outcome])

        prompt_parts.append("\n\nReturn JSON evaluation only.")

        result = proxy.initiate_chat(
            message="\n".join(prompt_parts),
            recipient=evaluator,
            summary_method="last_msg",
            max_turns=1,
        )

        elapsed_ms = (time.time() - start) * 1000
        return self._parse_result(result.summary, elapsed_ms)

    def _parse_result(self, raw: str, elapsed_ms: float) -> EvaluationResult:
        """Parse LLM output into EvaluationResult."""
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            return EvaluationResult(
                rationale="Failed to parse evaluation response",
                processing_time_ms=elapsed_ms,
            )

        try:
            data = json.loads(json_match.group())
            return EvaluationResult(
                relevance=float(data.get("relevance", 0)),
                faithfulness=float(data.get("faithfulness", 0)),
                helpfulness=float(data.get("helpfulness", 0)),
                tone_appropriateness=float(data.get("tone_appropriateness", 0)),
                completeness=float(data.get("completeness", 0)),
                accuracy=float(data.get("accuracy", 0)),
                clarity=float(data.get("clarity", 0)),
                actionability=float(data.get("actionability", 0)),
                contains_hallucination=bool(data.get("contains_hallucination", False)),
                escalation_appropriate=bool(data.get("escalation_appropriate", True)),
                pii_exposed=bool(data.get("pii_exposed", False)),
                rationale=str(data.get("rationale", "")),
                suggestions=list(data.get("suggestions", [])),
                processing_time_ms=elapsed_ms,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return EvaluationResult(
                rationale=f"Parse error: {e}",
                processing_time_ms=elapsed_ms,
            )


def evaluate_response_quality(
    query: str,
    response: str,
    context: str = "",
    llm_config: dict | None = None,
) -> EvaluationResult:
    """Convenience function to evaluate a single response."""
    evaluator = ResponseEvaluator(llm_config)
    return evaluator.evaluate(query, response, context)


# Lightweight rule-based checks (no LLM required)
def check_pii_exposure(text: str) -> list[str]:
    """Check for potential PII exposure in text."""
    patterns = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    }
    found = []
    for pii_type, pattern in patterns.items():
        if re.search(pattern, text):
            found.append(pii_type)
    return found


def check_response_length(response: str, min_words: int = 10, max_words: int = 500) -> dict:
    """Check if response length is appropriate."""
    word_count = len(response.split())
    return {
        "word_count": word_count,
        "too_short": word_count < min_words,
        "too_long": word_count > max_words,
        "appropriate": min_words <= word_count <= max_words,
    }


def check_sentiment(text: str) -> str:
    """Simple sentiment check based on keywords."""
    negative_words = {"angry", "frustrated", "terrible", "awful", "hate", "worst", "unacceptable"}
    positive_words = {"thank", "great", "excellent", "helpful", "appreciate", "resolved"}

    text_lower = text.lower()
    neg_count = sum(1 for word in negative_words if word in text_lower)
    pos_count = sum(1 for word in positive_words if word in text_lower)

    if neg_count > pos_count:
        return "negative"
    elif pos_count > neg_count:
        return "positive"
    return "neutral"
