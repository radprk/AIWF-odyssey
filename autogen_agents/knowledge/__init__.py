"""
Enhanced Knowledge Base System for Customer Support

This module provides a robust knowledge retrieval system that handles:
- Multiple document formats (PDF, TXT, Markdown, JSON, CSV)
- Different knowledge categories (policies, procedures, FAQs, products)
- Structured data (fee schedules, product details)
- Metadata-based filtering
- Hybrid search (semantic + keyword)
"""

from .document_loader import (
    KnowledgeDocument,
    load_documents_from_directory,
    load_pdf,
    load_text,
    load_json_knowledge,
    load_csv_knowledge,
)
from .knowledge_base import (
    KnowledgeBase,
    KnowledgeCategory,
    SearchResult,
)
from .retriever import (
    KnowledgeRetriever,
    retrieve_context,
)

__all__ = [
    "KnowledgeDocument",
    "load_documents_from_directory",
    "load_pdf",
    "load_text",
    "load_json_knowledge",
    "load_csv_knowledge",
    "KnowledgeBase",
    "KnowledgeCategory",
    "SearchResult",
    "KnowledgeRetriever",
    "retrieve_context",
]
