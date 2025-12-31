from __future__ import annotations

import random


SCENARIOS = {
    "billing": [
        "I was charged a {amount} fee I don't recognize.",
        "My bill increased by {amount} this month. Why?",
        "Please refund the {amount} overdraft fee.",
    ],
    "account": [
        "I need to reset my password but the link expired.",
        "Can you close my account and send confirmation?",
        "My account is locked after too many login attempts.",
    ],
    "transaction": [
        "Why is my transfer pending for {days} days?",
        "I made a deposit but it hasn't shown up.",
        "There is a duplicate charge on my card.",
    ],
    "complaint": [
        "I'm unhappy with how my dispute was handled.",
        "I filed a complaint last week and haven't heard back.",
        "Your service has been unreliable and I want to escalate.",
    ],
    "product info": [
        "What are the limits on international transfers?",
        "How does your premium account work?",
        "Do you offer purchase protection?",
    ],
    "sales": [
        "Can I upgrade to the premium plan?",
        "I'd like to add another line of service.",
        "What promotions are available this month?",
    ],
}


def generate_queries(count: int = 10, seed: int | None = None) -> list[str]:
    rng = random.Random(seed)
    categories = list(SCENARIOS.keys())
    queries = []
    for _ in range(count):
        category = rng.choice(categories)
        template = rng.choice(SCENARIOS[category])
        query = template.format(
            amount=rng.choice(["$15", "$32", "$120"]),
            days=rng.choice(["2", "5", "7"]),
        )
        queries.append(query)
    return queries
