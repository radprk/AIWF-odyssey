"""Knowledge base with vector storage and hybrid search."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .document_loader import KnowledgeDocument, load_documents_from_directory


class KnowledgeCategory(Enum):
    """Categories of knowledge documents."""
    POLICY = "policy"
    PROCEDURE = "procedure"
    FAQ = "faq"
    PRODUCT = "product"
    FEE_SCHEDULE = "fee_schedule"
    ARTICLE = "article"
    GUIDE = "guide"
    GENERAL = "general"


@dataclass
class SearchResult:
    """A search result with relevance score."""
    document: KnowledgeDocument
    score: float
    match_type: str = "semantic"  # "semantic", "keyword", "hybrid"

    def to_dict(self) -> dict:
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "match_type": self.match_type,
        }


class KnowledgeBase:
    """
    Vector-based knowledge base with hybrid search capabilities.

    Features:
    - Multiple document formats
    - Category-based filtering
    - Semantic search (via embeddings)
    - Keyword search fallback
    - Metadata filtering
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str | Path | None = None,
    ):
        self.embedding_model = embedding_model
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.documents: list[KnowledgeDocument] = []
        self._vector_store = None
        self._embeddings = None

    def _get_embeddings(self):
        """Lazy-load embeddings model."""
        if self._embeddings is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
            )
        return self._embeddings

    def add_documents(self, documents: list[KnowledgeDocument]) -> None:
        """Add documents to the knowledge base."""
        self.documents.extend(documents)
        self._rebuild_index()

    def add_document(self, document: KnowledgeDocument) -> None:
        """Add a single document."""
        self.documents.append(document)
        self._rebuild_index()

    def load_directory(self, directory: str | Path, recursive: bool = True) -> int:
        """Load all documents from a directory."""
        docs = load_documents_from_directory(directory, recursive)
        self.documents.extend(docs)
        self._rebuild_index()
        return len(docs)

    def _rebuild_index(self) -> None:
        """Rebuild the vector index."""
        if not self.documents:
            self._vector_store = None
            return

        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        # Convert to LangChain documents
        lc_docs = [
            Document(
                page_content=doc.content,
                metadata={
                    "id": doc.id,
                    "category": doc.category,
                    "source": doc.source,
                    "title": doc.title,
                    **doc.metadata,
                },
            )
            for doc in self.documents
        ]

        self._vector_store = FAISS.from_documents(lc_docs, self._get_embeddings())

        # Persist if directory specified
        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._vector_store.save_local(str(self.persist_directory / "faiss_index"))
            self._save_documents()

    def _save_documents(self) -> None:
        """Save documents metadata to disk."""
        if not self.persist_directory:
            return

        docs_path = self.persist_directory / "documents.json"
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in self.documents], f, indent=2)

    def _load_persisted(self) -> bool:
        """Load persisted index and documents."""
        if not self.persist_directory:
            return False

        index_path = self.persist_directory / "faiss_index"
        docs_path = self.persist_directory / "documents.json"

        if not index_path.exists() or not docs_path.exists():
            return False

        try:
            from langchain_community.vectorstores import FAISS

            self._vector_store = FAISS.load_local(
                str(index_path),
                self._get_embeddings(),
                allow_dangerous_deserialization=True,
            )

            with open(docs_path, "r", encoding="utf-8") as f:
                docs_data = json.load(f)

            self.documents = [
                KnowledgeDocument(
                    id=d["id"],
                    content=d["content"],
                    category=d["category"],
                    source=d["source"],
                    title=d.get("title", ""),
                    metadata=d.get("metadata", {}),
                    last_updated=d.get("last_updated", ""),
                    chunk_index=d.get("chunk_index", 0),
                    total_chunks=d.get("total_chunks", 1),
                )
                for d in docs_data
            ]

            return True
        except Exception as e:
            print(f"Warning: Failed to load persisted index: {e}")
            return False

    def search(
        self,
        query: str,
        k: int = 5,
        categories: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            k: Number of results to return
            categories: Filter by categories (e.g., ["faq", "policy"])
            min_score: Minimum relevance score (0-1)

        Returns:
            List of SearchResult objects
        """
        if not self._vector_store:
            return self._keyword_search(query, k, categories)

        # Semantic search
        results = self._vector_store.similarity_search_with_score(query, k=k * 2)

        search_results = []
        for doc, score in results:
            # FAISS returns distance, convert to similarity
            similarity = 1 / (1 + score)

            if similarity < min_score:
                continue

            # Filter by category
            if categories and doc.metadata.get("category") not in categories:
                continue

            # Find original document
            original = self._find_document(doc.metadata.get("id"))
            if original:
                search_results.append(SearchResult(
                    document=original,
                    score=similarity,
                    match_type="semantic",
                ))

        return search_results[:k]

    def _keyword_search(
        self,
        query: str,
        k: int,
        categories: list[str] | None = None,
    ) -> list[SearchResult]:
        """Fallback keyword search when no vector store."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for doc in self.documents:
            if categories and doc.category not in categories:
                continue

            content_lower = doc.content.lower()

            # Simple scoring: count matching words
            matches = sum(1 for word in query_words if word in content_lower)
            if matches > 0:
                score = matches / len(query_words)
                scored.append((doc, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(document=doc, score=score, match_type="keyword")
            for doc, score in scored[:k]
        ]

    def _find_document(self, doc_id: str) -> KnowledgeDocument | None:
        """Find document by ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def get_by_category(self, category: str) -> list[KnowledgeDocument]:
        """Get all documents in a category."""
        return [d for d in self.documents if d.category == category]

    def get_categories(self) -> list[str]:
        """Get all unique categories."""
        return list(set(d.category for d in self.documents))

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        categories = {}
        for doc in self.documents:
            categories[doc.category] = categories.get(doc.category, 0) + 1

        return {
            "total_documents": len(self.documents),
            "by_category": categories,
            "unique_sources": len(set(d.source for d in self.documents)),
            "has_vector_index": self._vector_store is not None,
        }

    def clear(self) -> None:
        """Clear all documents."""
        self.documents = []
        self._vector_store = None
