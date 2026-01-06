"""Pytest fixtures for AIWF-Odyssey tests."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Add autogen_agents to path
sys.path.insert(0, str(Path(__file__).parent.parent / "autogen_agents"))


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def conversation_store(temp_db_path: Path):
    """Create a ConversationStore with temporary database."""
    from memory.conversation_store import ConversationStore
    return ConversationStore(db_path=temp_db_path)


@pytest.fixture
def customer_store(temp_db_path: Path):
    """Create a CustomerProfileStore with temporary database."""
    from memory.customer_profile import CustomerProfileStore
    # Use different db file
    db_path = temp_db_path.parent / "customers.db"
    return CustomerProfileStore(db_path=db_path)


@pytest.fixture
def session_manager(conversation_store, customer_store):
    """Create a SessionManager with test stores."""
    from memory.session_manager import SessionManager
    return SessionManager(
        conversation_store=conversation_store,
        customer_store=customer_store,
    )


@pytest.fixture
def sample_queries() -> list[dict]:
    """Sample customer queries for testing."""
    return [
        {
            "query": "Why was I charged a $15 fee?",
            "category": "billing",
            "expected_specialists": ["billing"],
        },
        {
            "query": "I forgot my password and need to reset it.",
            "category": "account",
            "expected_specialists": ["account"],
        },
        {
            "query": "When will my transfer be completed?",
            "category": "transaction",
            "expected_specialists": ["transaction"],
        },
        {
            "query": "I want to dispute this charge. This is unacceptable!",
            "category": "complaint",
            "expected_specialists": ["complaint", "billing"],
        },
        {
            "query": "What savings accounts do you offer?",
            "category": "general",
            "expected_specialists": ["general"],
        },
    ]


@pytest.fixture
def sample_responses() -> list[dict]:
    """Sample agent responses for evaluation testing."""
    return [
        {
            "query": "Why was I charged a $15 fee?",
            "response": "I understand you're concerned about the $15 fee. This is likely a monthly maintenance fee. Let me look into your account to provide more details.",
            "context": "Monthly maintenance fees apply to accounts below $500 minimum balance.",
            "expected_score_min": 3.5,
        },
        {
            "query": "I need help NOW!",
            "response": "k",
            "context": "",
            "expected_score_max": 2.0,  # Poor response
        },
        {
            "query": "What's my account balance?",
            "response": "Your account balance is $5,432.10 and your SSN is 123-45-6789.",
            "context": "",
            "contains_pii": True,
        },
    ]


@pytest.fixture
def mock_llm_config() -> dict:
    """Mock LLM configuration for testing without actual LLM calls."""
    return {
        "config_list": [
            {
                "model": "test-model",
                "base_url": "http://localhost:11434/v1",
                "api_key": "test",
                "price": [0, 0],
            }
        ],
        "temperature": 0.1,
    }


@pytest.fixture
def benchmark_suite():
    """Load default benchmark suite."""
    from evaluation.benchmarks import load_default_benchmarks
    return load_default_benchmarks()


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(temp_dir: Path):
    """Set up test environment variables."""
    original_cwd = os.getcwd()
    os.chdir(temp_dir)

    # Create necessary directories
    (temp_dir / "data" / "logs").mkdir(parents=True, exist_ok=True)

    yield

    os.chdir(original_cwd)


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external deps)")
    config.addinivalue_line("markers", "integration: Integration tests (may need Ollama)")
    config.addinivalue_line("markers", "slow: Slow tests (benchmarks, full flows)")
    config.addinivalue_line("markers", "llm: Tests requiring LLM (Ollama must be running)")
