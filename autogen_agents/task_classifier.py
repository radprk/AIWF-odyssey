import json
from pathlib import Path

# Load O*NET task-to-estimate mapping
ONET_PATH = Path("data/onet/callcenter.json")
ONET_PATH = Path(__file__).parent.parent / "data" / "onet" / "callcenter.json"
with open(ONET_PATH, "r", encoding="utf-8") as f:
    onet_data = json.load(f)

TASK_KEYWORDS = {
    "billing": ["refund", "fee", "charge", "billing", "adjust bill", "dispute"],
    "account": ["cancel", "close", "restriction", "password", "account"],
    "transaction": ["transfer", "deposit", "withdraw", "funds", "pending", "sale"],
    "complaint": ["complaint", "issue", "grievance", "problem"],
    "product info": ["information", "details", "service", "policy", "insurance"],
    "sales": ["buy", "purchase", "order", "new service"],
}

ESTIMATES_SEC = {
    "billing": 240,
    "account": 180,
    "transaction": 210,
    "complaint": 300,
    "product info": 150,
    "sales": 200,
}

def classify(query):
    q = query.lower()
    for category, keywords in TASK_KEYWORDS.items():
        if any(word in q for word in keywords):
            return category.title(), ESTIMATES_SEC[category]
    return "General Inquiry", 180
