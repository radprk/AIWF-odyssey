# Local Development Guide for AIWF-Odyssey

This guide covers setting up the multi-agent customer support simulation locally on Cursor, with focus on persistent memory, enhanced evaluation, and testing.

---

## Prerequisites

### 1. Install Ollama (Local LLM Backend)

The project uses Ollama to run LLMs locally (no API keys needed).

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai/download

**Pull required models:**
```bash
ollama pull llama3
ollama pull mistral
ollama pull gemma
```

**Verify Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

### 2. Python Environment

```bash
# Navigate to project
cd AIWF-odyssey

# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install test dependencies (will be added)
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

---

## Project Structure (Multi-Agent Focus)

```
autogen_agents/
├── main.py              # CLI entry point
├── base_agents.py       # Agent definitions (L1/L2/L3 + specialists)
├── sim_agents.py        # TaskAwareAgent wrappers
├── router.py            # Query routing logic
├── task_classifier.py   # Task type classification
├── simulation_engine.py # Query handling orchestration
├── memory.py            # Logging (TO BE ENHANCED)
├── judge.py             # Response evaluation (TO BE ENHANCED)
├── metrics.py           # Performance metrics
├── retrieval.py         # FAQ RAG retrieval
├── faq_loader.py        # FAQ database loading
├── synthetic_data.py    # Test data generation
└── voice/tts.py         # Text-to-speech (optional)

tests/                   # TO BE CREATED
├── conftest.py          # Pytest fixtures
├── test_memory.py       # Memory system tests
├── test_evaluation.py   # Evaluation tests
├── test_agents.py       # Agent behavior tests
├── test_routing.py      # Routing tests
└── test_integration.py  # End-to-end tests
```

---

## Running the Application

### Interactive CLI Mode
```bash
cd autogen_agents
python main.py --mode cli
```

### With Specialist Routing
```bash
python main.py --mode cli --router
```

### With LLM-based Routing
```bash
python main.py --mode cli --router --llm-router
```

### Batch Demo (4 predefined queries)
```bash
python main.py --mode batch --metrics
```

### Synthetic Workload Test
```bash
python main.py --mode synthetic --synthetic-count 10 --metrics
```

### View Logs Dashboard
```bash
streamlit run streamlit_flow.py
```

---

## Current State Analysis

### Memory System (memory.py)
**Current:** Simple append-only JSONL logging per agent
- No conversation history tracking
- No customer profile persistence
- No cross-session context
- In-memory dict cleared on restart

### Evaluation System (judge.py)
**Current:** Basic 3-metric LLM judge
- Relevance (1-5)
- Faithfulness (1-5)
- Helpfulness (1-5)
- No automated benchmarking
- No regression detection

### Testing
**Current:** No tests exist

---

## Enhancement Plan

### 1. Persistent Memory System

**New Components:**
- `memory/conversation_store.py` - SQLite-based conversation history
- `memory/customer_profile.py` - Customer context persistence
- `memory/session_manager.py` - Session state management

**Features:**
- Cross-session conversation continuity
- Customer profile tracking (preferences, history, sentiment)
- Semantic search over past interactions
- Configurable retention policies

### 2. Enhanced Evaluation System

**New Components:**
- `evaluation/metrics.py` - Extended metrics (tone, accuracy, resolution)
- `evaluation/benchmarks.py` - Benchmark query sets with expected outcomes
- `evaluation/regression.py` - Quality regression detection
- `evaluation/reports.py` - Evaluation report generation

**New Metrics:**
- Response accuracy (fact-checking against FAQ)
- Tone appropriateness (professional, empathetic)
- Resolution completeness
- Escalation appropriateness
- Response latency analysis

### 3. Testing Suite

**Test Categories:**

**Unit Tests:**
- Memory operations (CRUD, search)
- Evaluation metric calculations
- Task classification accuracy
- Routing logic correctness

**Integration Tests:**
- Agent flow (L1→L2→L3 escalation)
- RAG retrieval quality
- End-to-end query handling

**Benchmark Tests:**
- Response quality regression
- Performance benchmarks
- Cost tracking accuracy

---

## Development Workflow in Cursor

### 1. Open Project
```
File → Open Folder → Select AIWF-odyssey
```

### 2. Configure Python Interpreter
- Cmd/Ctrl + Shift + P → "Python: Select Interpreter"
- Choose `./venv/bin/python`

### 3. Install Extensions (Recommended)
- Python
- Pylance
- Python Test Explorer

### 4. Running Tests (after test suite is created)
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=autogen_agents --cov-report=html

# Run specific test file
pytest tests/test_memory.py -v

# Run tests matching pattern
pytest tests/ -k "test_conversation" -v
```

### 5. Debugging
Add to `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run CLI",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/autogen_agents/main.py",
      "args": ["--mode", "cli", "--router"],
      "cwd": "${workspaceFolder}/autogen_agents"
    },
    {
      "name": "Run Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v"]
    }
  ]
}
```

---

## Environment Variables

Create `.env` in project root (optional):
```bash
# Ollama settings (defaults shown)
OLLAMA_BASE_URL=http://localhost:11434/v1

# Database paths
MEMORY_DB_PATH=data/memory.db
CONVERSATION_DB_PATH=data/conversations.db

# Logging
LOG_LEVEL=INFO
LOG_DIR=data/logs

# Testing
TEST_MODE=false
```

---

## Quick Start Checklist

- [ ] Ollama installed and running
- [ ] Required models pulled (llama3, mistral, gemma)
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Test dependencies installed (`pip install pytest pytest-asyncio pytest-cov pytest-mock`)
- [ ] Verify setup: `python main.py --mode batch --metrics`

---

## Troubleshooting

### Ollama Connection Error
```
Error: Could not connect to Ollama
```
**Fix:** Ensure Ollama is running: `ollama serve`

### Model Not Found
```
Error: model 'llama3' not found
```
**Fix:** Pull the model: `ollama pull llama3`

### Import Errors
```
ModuleNotFoundError: No module named 'autogen'
```
**Fix:** Install AG2: `pip install ag2[openai]>=0.4.0`

### FAISS/Embeddings Error
```
Error loading sentence transformers
```
**Fix:**
```bash
pip install sentence-transformers faiss-cpu
```
