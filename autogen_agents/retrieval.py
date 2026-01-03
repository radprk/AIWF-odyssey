from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from faq_loader import load_faq_database


@lru_cache(maxsize=1)
def _get_faq_db():
    faq_path = Path(__file__).parent.parent / "data" / "grounding" / "faq.txt"
    return load_faq_database(faq_path)


def _extract_keywords(query: str) -> set[str]:
    words = [word.strip(".,!?\"'():;").lower() for word in query.split()]
    return {word for word in words if len(word) >= 4}


def retrieve_context(query: str, k: int = 3) -> str:
    faq_db = _get_faq_db()
    retrieved_docs = faq_db.similarity_search(query, k=k)
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
