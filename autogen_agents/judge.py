from __future__ import annotations

import json
from dataclasses import dataclass

from autogen import AssistantAgent, UserProxyAgent


JUDGE_SYS_MSG = """
You are an evaluator for customer support responses.
Score the response on a 1-5 scale for:
- relevance
- faithfulness (uses provided context, avoids hallucinations)
- helpfulness

Return strict JSON with keys: relevance, faithfulness, helpfulness, rationale.
"""


@dataclass
class JudgeResult:
    relevance: int
    faithfulness: int
    helpfulness: int
    rationale: str

    def to_dict(self) -> dict:
        return {
            "relevance": self.relevance,
            "faithfulness": self.faithfulness,
            "helpfulness": self.helpfulness,
            "rationale": self.rationale,
        }


def _parse_json(raw: str) -> JudgeResult:
    try:
        payload = json.loads(raw)
        return JudgeResult(
            relevance=int(payload.get("relevance", 0)),
            faithfulness=int(payload.get("faithfulness", 0)),
            helpfulness=int(payload.get("helpfulness", 0)),
            rationale=str(payload.get("rationale", "")),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return JudgeResult(relevance=0, faithfulness=0, helpfulness=0, rationale="parse_error")


def evaluate_response(
    query: str,
    response: str,
    context: str,
    llm_config: dict,
) -> JudgeResult:
    judge_agent = AssistantAgent("Response_Judge", system_message=JUDGE_SYS_MSG, llm_config=llm_config)
    judge_proxy = UserProxyAgent(
        name="JudgeProxy",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
    )
    prompt = (
        "User query:\n"
        f"{query}\n\n"
        "Retrieved context:\n"
        f"{context}\n\n"
        "Response to evaluate:\n"
        f"{response}\n\n"
        "Return JSON only."
    )
    result = judge_proxy.initiate_chat(
        message=prompt,
        recipient=judge_agent,
        summary_method="last_msg",
        max_turns=1,
    )
    return _parse_json(result.summary)
