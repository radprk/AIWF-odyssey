"""High-level retriever interface for agent context injection."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .knowledge_base import KnowledgeBase, SearchResult


class KnowledgeRetriever:
    """
    High-level retriever for agent context injection.

    Provides formatted context strings suitable for LLM prompts.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        data_directory: str | Path | None = None,
    ):
        self.kb = knowledge_base or KnowledgeBase()

        if data_directory:
            self.kb.load_directory(data_directory)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        categories: list[str] | None = None,
        include_metadata: bool = True,
    ) -> list[SearchResult]:
        """Retrieve relevant documents."""
        return self.kb.search(query, k=k, categories=categories)

    def retrieve_context(
        self,
        query: str,
        k: int = 3,
        categories: list[str] | None = None,
        format: str = "plain",  # "plain", "markdown", "structured"
    ) -> str:
        """
        Retrieve context formatted for LLM prompt injection.

        Args:
            query: The user's query
            k: Number of documents to retrieve
            categories: Filter by categories
            format: Output format

        Returns:
            Formatted context string
        """
        results = self.retrieve(query, k=k, categories=categories)

        if not results:
            return ""

        if format == "markdown":
            return self._format_markdown(results)
        elif format == "structured":
            return self._format_structured(results)
        else:
            return self._format_plain(results)

    def _format_plain(self, results: list[SearchResult]) -> str:
        """Format as plain text."""
        lines = []
        for r in results:
            lines.append(r.document.content)
            lines.append("")
        return "\n".join(lines).strip()

    def _format_markdown(self, results: list[SearchResult]) -> str:
        """Format as markdown with headers."""
        lines = ["## Relevant Knowledge\n"]
        for i, r in enumerate(results, 1):
            doc = r.document
            lines.append(f"### {i}. {doc.title}")
            lines.append(f"*Category: {doc.category} | Relevance: {r.score:.2f}*\n")
            lines.append(doc.content)
            lines.append("")
        return "\n".join(lines)

    def _format_structured(self, results: list[SearchResult]) -> str:
        """Format as structured context with clear sections."""
        lines = ["=== KNOWLEDGE BASE CONTEXT ===\n"]

        # Group by category
        by_category: dict[str, list[SearchResult]] = {}
        for r in results:
            cat = r.document.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)

        for category, cat_results in by_category.items():
            lines.append(f"[{category.upper()}]")
            for r in cat_results:
                lines.append(f"- {r.document.title}:")
                lines.append(f"  {r.document.content}")
                lines.append("")

        lines.append("=== END CONTEXT ===")
        return "\n".join(lines)

    def get_policy_context(self, topic: str, k: int = 2) -> str:
        """Get policy-specific context."""
        return self.retrieve_context(topic, k=k, categories=["policy"], format="structured")

    def get_procedure_context(self, topic: str, k: int = 2) -> str:
        """Get procedure-specific context."""
        return self.retrieve_context(topic, k=k, categories=["procedure"], format="structured")

    def get_faq_context(self, query: str, k: int = 3) -> str:
        """Get FAQ-specific context."""
        return self.retrieve_context(query, k=k, categories=["faq"], format="plain")

    def get_fee_context(self, query: str, k: int = 2) -> str:
        """Get fee schedule context."""
        return self.retrieve_context(query, k=k, categories=["fee_schedule"], format="structured")

    def get_comprehensive_context(
        self,
        query: str,
        include_policies: bool = True,
        include_faqs: bool = True,
        include_procedures: bool = False,
    ) -> str:
        """
        Get comprehensive context from multiple categories.

        This is what you'd typically inject into an agent's prompt.
        """
        sections = []

        if include_faqs:
            faq_ctx = self.get_faq_context(query, k=2)
            if faq_ctx:
                sections.append("## FAQ Context\n" + faq_ctx)

        if include_policies:
            policy_ctx = self.get_policy_context(query, k=1)
            if policy_ctx:
                sections.append("## Policy Context\n" + policy_ctx)

        if include_procedures:
            proc_ctx = self.get_procedure_context(query, k=1)
            if proc_ctx:
                sections.append("## Procedure Context\n" + proc_ctx)

        return "\n\n".join(sections) if sections else ""


# Global retriever instance (lazy-loaded)
_default_retriever: KnowledgeRetriever | None = None


def get_default_retriever() -> KnowledgeRetriever:
    """Get or create the default retriever."""
    global _default_retriever

    if _default_retriever is None:
        # Default data directory
        data_dir = Path(__file__).parent.parent.parent / "data" / "grounding"
        _default_retriever = KnowledgeRetriever(data_directory=data_dir)

    return _default_retriever


def retrieve_context(query: str, k: int = 3) -> str:
    """
    Convenience function for backward compatibility.

    Replaces the old retrieval.py function.
    """
    retriever = get_default_retriever()
    return retriever.retrieve_context(query, k=k, format="plain")


def retrieve_comprehensive_context(query: str) -> str:
    """Get comprehensive context from all sources."""
    retriever = get_default_retriever()
    return retriever.get_comprehensive_context(query)
