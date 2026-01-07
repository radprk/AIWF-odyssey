from __future__ import annotations

import json
import re
from typing import Iterable, TYPE_CHECKING

# Lazy import for autogen to allow demo usage without full installation
if TYPE_CHECKING:
    from autogen import AssistantAgent, UserProxyAgent

from task_classifier import classify


ROUTER_SYS_MSG = """
You are a routing agent for a call center.
Your job is to decide which specialist agent(s) should handle the user query.

Return JSON with keys:
- "specialists": list of strings (e.g., "billing", "account", "transaction", "complaint", "general")
Keep the list short (1-2 items).
"""


def get_router_agent(llm_config):
    """Get the router agent. Requires autogen to be installed."""
    from autogen import AssistantAgent
    return AssistantAgent("Router", system_message=ROUTER_SYS_MSG, llm_config=llm_config)


def _rule_based_route(task_type: str) -> list[str]:
    normalized = task_type.lower()
    if "billing" in normalized:
        return ["billing"]
    if "account" in normalized:
        return ["account"]
    if "transaction" in normalized:
        return ["transaction"]
    if "complaint" in normalized:
        return ["complaint"]
    if "product" in normalized or "information" in normalized:
        return ["general"]
    return ["general"]


def _parse_router_response(raw: str) -> list[str]:
    try:
        payload = json.loads(raw)
        specialists = payload.get("specialists", [])
        if isinstance(specialists, list):
            return [str(item).strip().lower() for item in specialists if str(item).strip()]
    except json.JSONDecodeError:
        pass
    match = re.search(r"specialists?:\s*\[([^\]]+)\]", raw, re.IGNORECASE)
    if match:
        items = [item.strip(" \"'").lower() for item in match.group(1).split(",")]
        return [item for item in items if item]
    return []


def route_query(
    query: str,
    llm_config: dict,
    customer_proxy: "UserProxyAgent",
    use_llm_router: bool = False,
) -> tuple[list[str], str, int]:
    task_type, est_sec = classify(query)
    if not use_llm_router:
        return _rule_based_route(task_type), task_type, est_sec

    router_agent = get_router_agent(llm_config)
    result = customer_proxy.initiate_chat(
        message=query,
        recipient=router_agent,
        summary_method="last_msg",
        max_turns=1,
    )
    specialists = _parse_router_response(result.summary)
    if not specialists:
        specialists = _rule_based_route(task_type)
    return specialists, task_type, est_sec


def normalize_specialists(specialists: Iterable[str]) -> list[str]:
    return [spec.strip().lower() for spec in specialists if spec.strip()]
