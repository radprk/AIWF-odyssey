"""Tests for the enhanced knowledge base system."""

from __future__ import annotations

import json
import pytest
import sys
from pathlib import Path

# Add autogen_agents to path
sys.path.insert(0, str(Path(__file__).parent.parent / "autogen_agents"))


class TestDocumentLoader:
    """Tests for document loading functionality."""

    def test_load_text_file(self, temp_dir):
        """Test loading a simple text file."""
        from knowledge.document_loader import load_text

        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("This is test content.\nWith multiple lines.")

        docs = load_text(test_file, category="test")

        assert len(docs) >= 1
        assert docs[0].category == "test"
        assert "test content" in docs[0].content

    def test_load_text_with_chunking(self, temp_dir):
        """Test that large text files get chunked."""
        from knowledge.document_loader import load_text

        # Create large test file
        test_file = temp_dir / "large.txt"
        content = "This is a sentence. " * 100  # ~2000 chars
        test_file.write_text(content)

        docs = load_text(test_file, chunk_size=500)

        assert len(docs) > 1
        assert all(len(d.content) <= 600 for d in docs)  # Some overlap allowed

    def test_load_json_knowledge(self, temp_dir):
        """Test loading JSON knowledge file."""
        from knowledge.document_loader import load_json_knowledge

        # Create test JSON
        test_file = temp_dir / "knowledge.json"
        data = {
            "items": [
                {"title": "Item 1", "content": "Content for item 1"},
                {"title": "Item 2", "content": "Content for item 2"},
            ]
        }
        test_file.write_text(json.dumps(data))

        docs = load_json_knowledge(test_file)

        assert len(docs) == 2
        assert docs[0].title == "Item 1"
        assert "Content for item 1" in docs[0].content

    def test_load_csv_knowledge(self, temp_dir):
        """Test loading CSV knowledge file."""
        from knowledge.document_loader import load_csv_knowledge

        # Create test CSV
        test_file = temp_dir / "data.csv"
        test_file.write_text("""name,description,price
Product A,Description of A,$10
Product B,Description of B,$20""")

        docs = load_csv_knowledge(test_file, title_column="name")

        assert len(docs) == 2
        assert docs[0].title == "Product A"
        assert "Description of A" in docs[0].content

    def test_load_faq_text(self, temp_dir):
        """Test loading FAQ-style text file."""
        from knowledge.document_loader import load_faq_text

        # Create FAQ file
        test_file = temp_dir / "faq.txt"
        test_file.write_text("""Q: What is the return policy?
A: You can return items within 30 days.

Q: How do I contact support?
A: Email support@example.com or call 1-800-SUPPORT.""")

        docs = load_faq_text(test_file)

        assert len(docs) == 2
        assert "return policy" in docs[0].title.lower()
        assert "30 days" in docs[0].content

    def test_load_markdown(self, temp_dir):
        """Test loading Markdown file."""
        from knowledge.document_loader import load_markdown

        # Create MD file
        test_file = temp_dir / "guide.md"
        test_file.write_text("""# Getting Started Guide

## Step 1
Do this first.

## Step 2
Then do this.""")

        docs = load_markdown(test_file)

        assert len(docs) >= 1
        assert docs[0].title == "Getting Started Guide"

    def test_load_directory(self, temp_dir):
        """Test loading all documents from a directory."""
        from knowledge.document_loader import load_documents_from_directory

        # Create subdirectories with files
        (temp_dir / "faqs").mkdir()
        (temp_dir / "faqs" / "general.txt").write_text("""Q: Test question?
A: Test answer.""")

        (temp_dir / "policies").mkdir()
        (temp_dir / "policies" / "terms.md").write_text("# Terms of Service\nContent here.")

        docs = load_documents_from_directory(temp_dir)

        assert len(docs) >= 2
        categories = {d.category for d in docs}
        assert "faq" in categories or "policy" in categories


class TestKnowledgeBase:
    """Tests for KnowledgeBase class."""

    def test_create_knowledge_base(self):
        """Test creating an empty knowledge base."""
        from knowledge.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        assert kb.documents == []
        assert kb.get_stats()["total_documents"] == 0

    def test_add_documents(self):
        """Test adding documents to knowledge base."""
        from knowledge.knowledge_base import KnowledgeBase
        from knowledge.document_loader import KnowledgeDocument

        kb = KnowledgeBase()

        docs = [
            KnowledgeDocument(
                id="doc1",
                content="Content about billing fees",
                category="faq",
                source="test.txt",
                title="Billing FAQ",
            ),
            KnowledgeDocument(
                id="doc2",
                content="Content about account access",
                category="faq",
                source="test.txt",
                title="Account FAQ",
            ),
        ]

        kb.add_documents(docs)

        assert len(kb.documents) == 2
        assert kb.get_stats()["total_documents"] == 2

    def test_search_keyword_fallback(self):
        """Test keyword search when no vector store."""
        from knowledge.knowledge_base import KnowledgeBase
        from knowledge.document_loader import KnowledgeDocument

        kb = KnowledgeBase()
        kb._vector_store = None  # Force keyword search

        kb.documents = [
            KnowledgeDocument("1", "Information about fees and charges", "faq", "test", "Fees"),
            KnowledgeDocument("2", "How to reset your password", "faq", "test", "Password"),
            KnowledgeDocument("3", "Transfer money between accounts", "faq", "test", "Transfers"),
        ]

        results = kb.search("fees", k=2)

        assert len(results) >= 1
        assert "fee" in results[0].document.content.lower()

    def test_filter_by_category(self):
        """Test filtering search results by category."""
        from knowledge.knowledge_base import KnowledgeBase
        from knowledge.document_loader import KnowledgeDocument

        kb = KnowledgeBase()
        kb._vector_store = None

        kb.documents = [
            KnowledgeDocument("1", "Fee policy document", "policy", "test", "Fee Policy"),
            KnowledgeDocument("2", "Fee FAQ document", "faq", "test", "Fee FAQ"),
        ]

        # Search only policies
        results = kb.search("fee", k=5, categories=["policy"])

        assert len(results) == 1
        assert results[0].document.category == "policy"

    def test_get_by_category(self):
        """Test getting all documents by category."""
        from knowledge.knowledge_base import KnowledgeBase
        from knowledge.document_loader import KnowledgeDocument

        kb = KnowledgeBase()
        kb.documents = [
            KnowledgeDocument("1", "Content 1", "faq", "test", "FAQ 1"),
            KnowledgeDocument("2", "Content 2", "policy", "test", "Policy 1"),
            KnowledgeDocument("3", "Content 3", "faq", "test", "FAQ 2"),
        ]

        faqs = kb.get_by_category("faq")
        assert len(faqs) == 2

    def test_get_categories(self):
        """Test getting unique categories."""
        from knowledge.knowledge_base import KnowledgeBase
        from knowledge.document_loader import KnowledgeDocument

        kb = KnowledgeBase()
        kb.documents = [
            KnowledgeDocument("1", "C1", "faq", "test", "T1"),
            KnowledgeDocument("2", "C2", "policy", "test", "T2"),
            KnowledgeDocument("3", "C3", "procedure", "test", "T3"),
        ]

        categories = kb.get_categories()
        assert set(categories) == {"faq", "policy", "procedure"}


class TestKnowledgeRetriever:
    """Tests for KnowledgeRetriever class."""

    def test_retrieve_context_plain(self):
        """Test retrieving context in plain format."""
        from knowledge.knowledge_base import KnowledgeBase
        from knowledge.retriever import KnowledgeRetriever
        from knowledge.document_loader import KnowledgeDocument

        kb = KnowledgeBase()
        kb._vector_store = None
        kb.documents = [
            KnowledgeDocument("1", "The monthly fee is $15.", "faq", "test", "Fee Info"),
        ]

        retriever = KnowledgeRetriever(knowledge_base=kb)
        context = retriever.retrieve_context("monthly fee", format="plain")

        assert "monthly fee" in context.lower() or "$15" in context

    def test_retrieve_context_markdown(self):
        """Test retrieving context in markdown format."""
        from knowledge.knowledge_base import KnowledgeBase
        from knowledge.retriever import KnowledgeRetriever
        from knowledge.document_loader import KnowledgeDocument

        kb = KnowledgeBase()
        kb._vector_store = None
        kb.documents = [
            KnowledgeDocument("1", "Test content", "faq", "test", "Test Title"),
        ]

        retriever = KnowledgeRetriever(knowledge_base=kb)
        context = retriever.retrieve_context("test", format="markdown")

        assert "## Relevant Knowledge" in context
        assert "Test Title" in context

    def test_retrieve_context_structured(self):
        """Test retrieving context in structured format."""
        from knowledge.knowledge_base import KnowledgeBase
        from knowledge.retriever import KnowledgeRetriever
        from knowledge.document_loader import KnowledgeDocument

        kb = KnowledgeBase()
        kb._vector_store = None
        kb.documents = [
            KnowledgeDocument("1", "Policy content", "policy", "test", "Policy Doc"),
        ]

        retriever = KnowledgeRetriever(knowledge_base=kb)
        context = retriever.retrieve_context("policy", format="structured")

        assert "KNOWLEDGE BASE CONTEXT" in context
        assert "[POLICY]" in context

    def test_category_specific_retrieval(self):
        """Test category-specific retrieval methods."""
        from knowledge.knowledge_base import KnowledgeBase
        from knowledge.retriever import KnowledgeRetriever
        from knowledge.document_loader import KnowledgeDocument

        kb = KnowledgeBase()
        kb._vector_store = None
        kb.documents = [
            KnowledgeDocument("1", "Fee schedule info", "fee_schedule", "test", "Fees"),
            KnowledgeDocument("2", "Policy info", "policy", "test", "Policy"),
            KnowledgeDocument("3", "FAQ info", "faq", "test", "FAQ"),
        ]

        retriever = KnowledgeRetriever(knowledge_base=kb)

        faq_ctx = retriever.get_faq_context("info")
        policy_ctx = retriever.get_policy_context("info")
        fee_ctx = retriever.get_fee_context("info")

        # Each should only return relevant category
        assert "FAQ" in faq_ctx or "faq" in faq_ctx.lower() or faq_ctx == ""
        assert "Policy" in policy_ctx or "policy" in policy_ctx.lower() or policy_ctx == ""


class TestRealDataLoading:
    """Tests using actual data files."""

    def test_load_fee_schedule(self):
        """Test loading the fee schedule JSON."""
        from knowledge.document_loader import load_json_knowledge

        fee_path = Path(__file__).parent.parent / "data" / "grounding" / "fees" / "fee_schedule.json"
        if not fee_path.exists():
            pytest.skip("Fee schedule file not found")

        docs = load_json_knowledge(fee_path, content_field="conditions", title_field="fee_name")

        assert len(docs) > 0
        # Check that fee info is in content
        assert any("$" in d.content or "fee" in d.content.lower() for d in docs)

    def test_load_product_info(self):
        """Test loading product information."""
        from knowledge.document_loader import load_json_knowledge

        product_path = Path(__file__).parent.parent / "data" / "grounding" / "products" / "account_types.json"
        if not product_path.exists():
            pytest.skip("Product file not found")

        docs = load_json_knowledge(product_path)

        assert len(docs) > 0

    def test_load_procedures(self):
        """Test loading procedure documents."""
        from knowledge.document_loader import load_markdown

        proc_path = Path(__file__).parent.parent / "data" / "grounding" / "procedures" / "dispute_process.md"
        if not proc_path.exists():
            pytest.skip("Procedure file not found")

        docs = load_markdown(proc_path, category="procedure")

        assert len(docs) > 0
        assert "dispute" in docs[0].content.lower()

    def test_load_full_knowledge_directory(self):
        """Test loading entire knowledge directory."""
        from knowledge.document_loader import load_documents_from_directory

        data_dir = Path(__file__).parent.parent / "data" / "grounding"
        if not data_dir.exists():
            pytest.skip("Data directory not found")

        docs = load_documents_from_directory(data_dir)

        # Should load multiple documents from different categories
        assert len(docs) > 5
        categories = {d.category for d in docs}
        assert len(categories) >= 2  # At least 2 different categories


class TestSearchResult:
    """Tests for SearchResult class."""

    def test_search_result_to_dict(self):
        """Test SearchResult serialization."""
        from knowledge.knowledge_base import SearchResult
        from knowledge.document_loader import KnowledgeDocument

        doc = KnowledgeDocument("1", "Content", "faq", "test", "Title")
        result = SearchResult(document=doc, score=0.85, match_type="semantic")

        data = result.to_dict()

        assert data["score"] == 0.85
        assert data["match_type"] == "semantic"
        assert data["document"]["id"] == "1"
