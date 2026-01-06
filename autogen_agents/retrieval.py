from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from faq_loader import load_faq_database


@lru_cache(maxsize=1)
def _get_faq_db():
    faq_path = Path(__file__).parent.parent / "data" / "grounding" / "faq.txt"
    return load_faq_database(faq_path)


def retrieve_context(query: str, k: int = 3) -> str:
    faq_db = _get_faq_db()
    retrieved_docs = faq_db.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in retrieved_docs])
