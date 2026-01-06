from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from faq_loader import load_faq_database


_GROUNDING_DIR = Path(__file__).parent.parent / "data" / "grounding"


@lru_cache(maxsize=1)
def _get_general_db():
    return load_faq_database(_GROUNDING_DIR / "faq.txt")


@lru_cache(maxsize=1)
def _get_financial_db():
    return load_faq_database(_GROUNDING_DIR / "financial_faq.txt")


@lru_cache(maxsize=1)
def _get_healthcare_db():
    return load_faq_database(_GROUNDING_DIR / "healthcare_faq.txt")


_FINANCIAL_KEYWORDS = {
    "transfer", "ach", "wire", "transaction", "deposit", "withdrawal",
    "fee", "charge", "card", "account", "bank", "balance", "trade",
}
_HEALTHCARE_KEYWORDS = {
    "claim", "insurance", "billing", "medical", "patient", "records",
    "appointment", "provider", "diagnosis", "treatment",
}


def detect_industry(query: str) -> str:
    words = _extract_keywords(query)
    if words & _FINANCIAL_KEYWORDS:
        return "financial"
    if words & _HEALTHCARE_KEYWORDS:
        return "healthcare"
    return "general"


def _extract_keywords(query: str) -> set[str]:
    words = [word.strip(".,!?\"'():;").lower() for word in query.split()]
    return {word for word in words if len(word) >= 4}


def retrieve_context(query: str, k: int = 3) -> str:
    industry = detect_industry(query)
    if industry == "financial":
        faq_db = _get_financial_db()
    elif industry == "healthcare":
        faq_db = _get_healthcare_db()
    else:
        faq_db = _get_general_db()

    retrieved_docs = faq_db.similarity_search(query, k=k)
    if not retrieved_docs:
        retrieved_docs = _get_general_db().similarity_search(query, k=k)
        if not retrieved_docs:
            return ""

    keywords = _extract_keywords(query)
    if not keywords:
        return "\n".join([doc.page_content for doc in retrieved_docs])

    filtered = [
        doc for doc in retrieved_docs
        if any(keyword in doc.page_content.lower() for keyword in keywords)
    ]
    docs = filtered or retrieved_docs
    return "\n".join([doc.page_content for doc in docs])
