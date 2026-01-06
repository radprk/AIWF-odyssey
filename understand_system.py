"""
AIWF-Odyssey: Understanding the System
=======================================

Run this file to see how each component works:
    python understand_system.py

This will walk you through the key concepts with examples.
"""

import sys
from pathlib import Path

# Add autogen_agents to path
sys.path.insert(0, str(Path(__file__).parent / "autogen_agents"))


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def demo_task_classifier():
    """Show how queries get classified into categories."""
    print_header("1. TASK CLASSIFIER")
    print("""
The classifier looks at a customer query and decides:
- What TYPE of issue is this? (billing, account, transaction, etc.)
- How long will it take to resolve? (estimated seconds)

It uses simple keyword matching:
""")

    from task_classifier import classify, TASK_KEYWORDS

    print("Keywords per category:")
    for category, keywords in TASK_KEYWORDS.items():
        print(f"  {category}: {keywords}")

    print("\n--- Examples ---")
    test_queries = [
        "Why was I charged a $15 fee?",
        "I need to reset my password",
        "When will my transfer complete?",
        "I have a complaint about your service",
        "What insurance policies do you offer?",
        "Hello, I have a question",  # No keywords = General
    ]

    for query in test_queries:
        task_type, estimate_sec = classify(query)
        print(f"\n  Query: \"{query}\"")
        print(f"  → Type: {task_type}, Estimated time: {estimate_sec}s ({estimate_sec//60}m {estimate_sec%60}s)")


def demo_routing():
    """Show how queries get routed to specialists."""
    print_header("2. ROUTING")
    print("""
After classification, the query is ROUTED to the right specialist agent.
Each specialist has domain expertise:
- billing: handles fees, charges, refunds
- account: handles passwords, access, closures
- transaction: handles transfers, deposits
- complaint: handles escalated complaints
- general: handles everything else
""")

    from task_classifier import classify
    from router import _rule_based_route

    print("--- Routing Examples ---")
    queries = [
        "Why was I charged a $15 fee?",
        "I forgot my password",
        "My transfer is pending",
    ]

    for query in queries:
        task_type, _ = classify(query)
        specialists = _rule_based_route(task_type)
        print(f"\n  Query: \"{query}\"")
        print(f"  → Classified as: {task_type}")
        print(f"  → Routed to: {specialists}")


def demo_evaluation():
    """Show how responses get evaluated."""
    print_header("3. EVALUATION SYSTEM")
    print("""
When an agent responds, we evaluate the quality:

METRICS (scored 1-5):
- Relevance: Does it answer the actual question?
- Faithfulness: Does it stick to facts (no hallucinations)?
- Helpfulness: Does it actually help solve the problem?
- Tone: Is it professional and empathetic?
- Completeness: Does it fully address the issue?
- Accuracy: Are the facts correct?
- Clarity: Is it easy to understand?
- Actionability: Does it give clear next steps?

CHECKS (pass/fail):
- Hallucination: Did it make up information?
- PII Exposure: Did it leak personal data?
- Escalation: Was escalation decision appropriate?

The OVERALL SCORE is a weighted average with penalties.
""")

    from evaluation.metrics import EvaluationResult, check_pii_exposure, check_sentiment

    # Show a good response
    print("--- Example: Good Response ---")
    good = EvaluationResult(
        relevance=4.5,
        faithfulness=5.0,
        helpfulness=4.0,
        tone_appropriateness=4.5,
        completeness=4.0,
        accuracy=5.0,
        clarity=4.5,
        actionability=4.0,
    )
    print(f"  Scores: relevance={good.relevance}, helpfulness={good.helpfulness}, ...")
    print(f"  Overall Score: {good.overall_score}/5")
    print(f"  Passed: {good.passed}")

    # Show a bad response
    print("\n--- Example: Bad Response (with hallucination) ---")
    bad = EvaluationResult(
        relevance=3.0,
        faithfulness=2.0,
        helpfulness=2.5,
        tone_appropriateness=3.0,
        completeness=2.0,
        accuracy=2.0,
        clarity=3.0,
        actionability=2.0,
        contains_hallucination=True,  # 30% penalty!
    )
    print(f"  Scores: relevance={bad.relevance}, helpfulness={bad.helpfulness}, ...")
    print(f"  Contains hallucination: {bad.contains_hallucination}")
    print(f"  Overall Score: {bad.overall_score}/5 (penalized for hallucination)")
    print(f"  Passed: {bad.passed}")

    # Show PII detection
    print("\n--- PII Detection ---")
    texts = [
        "Your account balance is $500",
        "Your SSN is 123-45-6789",  # BAD!
        "Call us at 555-123-4567",  # Phone detected
    ]
    for text in texts:
        pii = check_pii_exposure(text)
        print(f"  \"{text}\"")
        print(f"  → PII found: {pii if pii else 'None'}")


def demo_benchmarks():
    """Show how benchmarks work for systematic testing."""
    print_header("4. BENCHMARK SYSTEM")
    print("""
Benchmarks are PRE-DEFINED test cases to evaluate agent performance:
- Each case has a query, expected behavior, and quality thresholds
- Run benchmarks to measure agent quality over time
- Detect regressions when you make changes
""")

    from evaluation.benchmarks import load_default_benchmarks

    suite = load_default_benchmarks()
    print(f"\nLoaded {len(suite.cases)} benchmark cases:")

    # Group by category
    by_category = {}
    for case in suite.cases:
        by_category.setdefault(case.category, []).append(case)

    for category, cases in by_category.items():
        print(f"\n  {category.upper()}:")
        for case in cases:
            escalate = "→ should escalate" if case.should_escalate else ""
            print(f"    - {case.name} ({case.difficulty}) {escalate}")


def demo_memory():
    """Show how persistent memory works."""
    print_header("5. PERSISTENT MEMORY")
    print("""
Memory allows the system to remember:
- Conversation history (what was discussed before)
- Customer profiles (who is this person, their history)
- Session state (current conversation context)

This enables:
- Personalized responses for returning customers
- Context-aware agent behavior
- Tracking customer sentiment over time
""")

    from memory import ConversationStore, CustomerProfileStore, SessionManager
    import tempfile

    # Use temp DB for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        conv_store = ConversationStore(f"{tmpdir}/conv.db")
        cust_store = CustomerProfileStore(f"{tmpdir}/cust.db")
        session_mgr = SessionManager(conv_store, cust_store)

        # Simulate a conversation
        print("\n--- Simulating a Customer Interaction ---")

        # Create session
        session = session_mgr.create_session("session-001", "customer-123")
        print(f"  Created session: {session.session_id}")

        # Start conversation
        session_mgr.start_conversation("session-001", "Why was I charged $15?")
        print("  Customer: 'Why was I charged $15?'")

        # Agent responds
        session_mgr.add_agent_response("session-001", "L1_Support",
            "I see a $15 maintenance fee. Let me look into this.")
        print("  L1_Support: 'I see a $15 maintenance fee...'")

        # Escalate
        session_mgr.record_escalation("session-001", "L1_Support", "L2_Support",
            "Complex fee dispute")
        print("  ↑ Escalated to L2_Support")

        # Resolve
        session_mgr.resolve_conversation("session-001", "Fee waived", sentiment=0.8)
        print("  ✓ Resolved: Fee waived, sentiment: positive")

        # Check customer profile
        profile = cust_store.get_customer("customer-123")
        print(f"\n  Customer Profile After Interaction:")
        print(f"    - Total interactions: {profile.total_interactions}")
        print(f"    - Escalation count: {profile.escalation_count}")
        print(f"    - Resolved count: {profile.resolved_count}")
        print(f"    - Sentiment: {profile.last_sentiment}")


def demo_flow():
    """Show the complete flow."""
    print_header("6. COMPLETE FLOW")
    print("""
Here's how everything connects:

    Customer Query
         ↓
    ┌────────────────┐
    │ Task Classifier │ → Determines type (billing, account, etc.)
    └────────────────┘
         ↓
    ┌────────────────┐
    │    Router      │ → Sends to right specialist(s)
    └────────────────┘
         ↓
    ┌────────────────┐
    │ Agent (L1/L2/L3)│ → Generates response (uses LLM)
    └────────────────┘
         ↓
    ┌────────────────┐
    │   Evaluator    │ → Scores response quality
    └────────────────┘
         ↓
    ┌────────────────┐
    │    Memory      │ → Saves conversation & updates profile
    └────────────────┘
         ↓
    Response to Customer
""")


def main():
    print("\n" + "=" * 60)
    print("  AIWF-ODYSSEY: UNDERSTANDING THE SYSTEM")
    print("=" * 60)
    print("""
This script walks you through how each component works.
No LLM required - just demonstrating the concepts.
""")

    demo_task_classifier()
    demo_routing()
    demo_evaluation()
    demo_benchmarks()
    demo_memory()
    demo_flow()

    print_header("WHAT TO DO NEXT")
    print("""
1. RUN THE MULTI-AGENT SYSTEM (requires Ollama):
   cd autogen_agents
   python main.py --mode cli --router

2. RUN BENCHMARKS (requires Ollama):
   # Coming soon - run evaluation suite against your agents

3. VIEW LOGS:
   streamlit run autogen_agents/streamlit_flow.py

4. RUN TESTS:
   pytest tests/ -v

5. EXTEND THE SYSTEM:
   - Add keywords to task_classifier.py for better classification
   - Add benchmark cases in evaluation/benchmarks.py
   - Modify agent prompts in base_agents.py
""")


if __name__ == "__main__":
    main()
