"""Document loaders for various file formats."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class KnowledgeDocument:
    """A document in the knowledge base."""
    id: str
    content: str
    category: str  # "policy", "procedure", "faq", "product", "fee_schedule", etc.
    source: str  # File path or URL
    title: str = ""

    # Metadata for filtering and ranking
    metadata: dict[str, Any] = field(default_factory=dict)

    # When was this document last updated?
    last_updated: str = ""

    # For chunked documents, track position
    chunk_index: int = 0
    total_chunks: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "source": self.source,
            "title": self.title,
            "metadata": self.metadata,
            "last_updated": self.last_updated,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


def _generate_id(source: str, chunk_index: int = 0) -> str:
    """Generate a unique document ID."""
    import hashlib
    hash_input = f"{source}:{chunk_index}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within last 100 chars
            for sep in [". ", ".\n", "? ", "! ", "\n\n"]:
                last_sep = text[max(start, end - 100):end].rfind(sep)
                if last_sep != -1:
                    end = max(start, end - 100) + last_sep + len(sep)
                    break

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]  # Filter empty


def load_text(
    path: Path | str,
    category: str = "general",
    chunk_size: int = 500,
    metadata: dict | None = None,
) -> list[KnowledgeDocument]:
    """Load a text file into knowledge documents."""
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = _chunk_text(content, chunk_size)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(KnowledgeDocument(
            id=_generate_id(str(path), i),
            content=chunk,
            category=category,
            source=str(path),
            title=path.stem.replace("_", " ").title(),
            metadata=metadata or {},
            last_updated=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            chunk_index=i,
            total_chunks=len(chunks),
        ))

    return docs


def load_pdf(
    path: Path | str,
    category: str = "policy",
    chunk_size: int = 500,
    metadata: dict | None = None,
) -> list[KnowledgeDocument]:
    """Load a PDF file into knowledge documents."""
    path = Path(path)

    try:
        # Try PyPDF2 first
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        content = "\n\n".join(pages)
    except ImportError:
        try:
            # Fallback to pdfplumber
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                content = "\n\n".join(pages)
        except ImportError:
            # No PDF library available
            print(f"Warning: No PDF library available. Install PyPDF2 or pdfplumber.")
            return []

    if not content.strip():
        return []

    chunks = _chunk_text(content, chunk_size)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(KnowledgeDocument(
            id=_generate_id(str(path), i),
            content=chunk,
            category=category,
            source=str(path),
            title=path.stem.replace("_", " ").title(),
            metadata=metadata or {},
            last_updated=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            chunk_index=i,
            total_chunks=len(chunks),
        ))

    return docs


def load_markdown(
    path: Path | str,
    category: str = "article",
    chunk_size: int = 500,
    metadata: dict | None = None,
) -> list[KnowledgeDocument]:
    """Load a Markdown file, preserving structure."""
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract title from first heading
    title = path.stem.replace("_", " ").title()
    lines = content.split("\n")
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    chunks = _chunk_text(content, chunk_size)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(KnowledgeDocument(
            id=_generate_id(str(path), i),
            content=chunk,
            category=category,
            source=str(path),
            title=title,
            metadata=metadata or {},
            last_updated=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            chunk_index=i,
            total_chunks=len(chunks),
        ))

    return docs


def load_json_knowledge(
    path: Path | str,
    category: str = "structured",
    content_field: str = "content",
    title_field: str = "title",
) -> list[KnowledgeDocument]:
    """
    Load structured knowledge from JSON.

    Expects either:
    - Array of objects with content_field
    - Object with nested items
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # Look for common array keys
        for key in ["items", "documents", "entries", "faqs", "articles"]:
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
        else:
            # Treat the whole dict as a single item
            items = [data]
    else:
        return []

    for i, item in enumerate(items):
        if isinstance(item, dict):
            content = item.get(content_field, "")
            title = item.get(title_field, f"Item {i+1}")

            # If no content field, serialize the whole item
            if not content:
                content = json.dumps(item, indent=2)

            # Extract any metadata
            meta = {k: v for k, v in item.items()
                    if k not in [content_field, title_field] and isinstance(v, (str, int, float, bool))}

            docs.append(KnowledgeDocument(
                id=_generate_id(str(path), i),
                content=str(content),
                category=category,
                source=str(path),
                title=str(title),
                metadata=meta,
                last_updated=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            ))
        elif isinstance(item, str):
            docs.append(KnowledgeDocument(
                id=_generate_id(str(path), i),
                content=item,
                category=category,
                source=str(path),
                title=f"Item {i+1}",
                last_updated=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            ))

    return docs


def load_csv_knowledge(
    path: Path | str,
    category: str = "data",
    content_columns: list[str] | None = None,
    title_column: str | None = None,
) -> list[KnowledgeDocument]:
    """
    Load structured knowledge from CSV.

    Each row becomes a document. Content is built from specified columns
    or all columns if not specified.
    """
    path = Path(path)

    docs = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            # Build content from specified columns or all
            if content_columns:
                content_parts = [f"{col}: {row.get(col, '')}" for col in content_columns if row.get(col)]
            else:
                content_parts = [f"{k}: {v}" for k, v in row.items() if v]

            content = "\n".join(content_parts)

            # Get title
            title = row.get(title_column, f"Row {i+1}") if title_column else f"Row {i+1}"

            # All columns as metadata
            meta = {k: v for k, v in row.items() if v}

            docs.append(KnowledgeDocument(
                id=_generate_id(str(path), i),
                content=content,
                category=category,
                source=str(path),
                title=str(title),
                metadata=meta,
                last_updated=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            ))

    return docs


def load_faq_text(
    path: Path | str,
    category: str = "faq",
) -> list[KnowledgeDocument]:
    """
    Load FAQ-style text file where each Q&A pair becomes a document.

    Expected format:
    Q: Question here?
    A: Answer here.

    Q: Another question?
    A: Another answer.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by Q: pattern
    import re
    qa_pattern = re.compile(r"Q:\s*(.+?)\s*\nA:\s*(.+?)(?=\nQ:|\Z)", re.DOTALL)
    matches = qa_pattern.findall(content)

    docs = []
    for i, (question, answer) in enumerate(matches):
        question = question.strip()
        answer = answer.strip()

        docs.append(KnowledgeDocument(
            id=_generate_id(str(path), i),
            content=f"Question: {question}\n\nAnswer: {answer}",
            category=category,
            source=str(path),
            title=question[:100] + "..." if len(question) > 100 else question,
            metadata={"question": question, "answer": answer},
            last_updated=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        ))

    return docs


def load_documents_from_directory(
    directory: Path | str,
    recursive: bool = True,
) -> list[KnowledgeDocument]:
    """
    Load all supported documents from a directory.

    Category is inferred from subdirectory name or file extension.
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    docs = []

    # Map extensions to loaders
    loaders = {
        ".txt": load_text,
        ".md": load_markdown,
        ".json": load_json_knowledge,
        ".csv": load_csv_knowledge,
        ".pdf": load_pdf,
    }

    # Map directory names to categories
    category_map = {
        "policies": "policy",
        "policy": "policy",
        "procedures": "procedure",
        "faqs": "faq",
        "faq": "faq",
        "products": "product",
        "fees": "fee_schedule",
        "articles": "article",
        "guides": "guide",
        "grounding": "faq",  # Legacy support
    }

    pattern = "**/*" if recursive else "*"

    for path in directory.glob(pattern):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        if ext not in loaders:
            continue

        # Infer category from parent directory
        parent_name = path.parent.name.lower()
        category = category_map.get(parent_name, "general")

        # Special handling for FAQ-style text files
        if ext == ".txt" and "faq" in path.stem.lower():
            docs.extend(load_faq_text(path, category="faq"))
        else:
            loader = loaders[ext]
            try:
                docs.extend(loader(path, category=category))
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

    return docs
