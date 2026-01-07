"""
Interactive Demo: Multi-Agent Customer Support System

Run with: streamlit run demo_app.py
"""

import streamlit as st
import sys
import time
from pathlib import Path
from datetime import datetime

# Add autogen_agents to path
sys.path.insert(0, str(Path(__file__).parent / "autogen_agents"))

# Page config
st.set_page_config(
    page_title="AI Customer Support Demo",
    page_icon="🎧",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
if "customer_id" not in st.session_state:
    st.session_state.customer_id = "demo-customer-001"
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "escalation_chain" not in st.session_state:
    st.session_state.escalation_chain = []


def classify_query(query: str) -> tuple[str, int]:
    """Classify the query and return task type + estimate."""
    try:
        from task_classifier import classify
        return classify(query)
    except Exception:
        return "General Inquiry", 180


def get_specialists(task_type: str) -> list[str]:
    """Get specialists for task type."""
    try:
        from router import _rule_based_route
        return _rule_based_route(task_type)
    except Exception:
        return ["general"]


def get_knowledge_context(query: str) -> str:
    """Get relevant knowledge context."""
    try:
        from knowledge import KnowledgeRetriever
        data_dir = Path(__file__).parent / "data" / "grounding"
        retriever = KnowledgeRetriever(data_directory=data_dir)
        return retriever.get_comprehensive_context(query)
    except Exception as e:
        return f"(Knowledge retrieval unavailable: {e})"


def simulate_agent_response(query: str, agent_name: str, context: str) -> tuple[str, float, bool]:
    """
    Simulate agent response.
    Returns: (response, cost, should_escalate)
    """
    # Agent hourly rates
    rates = {
        "L1_Support": 18.0,
        "L2_Support": 25.0,
        "L3_Expert": 40.0,
        "Billing_Specialist": 22.0,
        "Account_Specialist": 22.0,
    }

    # Simulate processing time (2-10 seconds)
    process_time = 3.0  # seconds
    hourly_rate = rates.get(agent_name, 20.0)
    cost = (process_time / 3600) * hourly_rate

    # Check for escalation triggers
    escalation_keywords = ["supervisor", "manager", "escalate", "unacceptable", "lawyer", "sue"]
    should_escalate = any(kw in query.lower() for kw in escalation_keywords)

    # Generate response based on context and query
    if "fee" in query.lower() and "15" in query:
        response = f"""I understand you're concerned about the $15 fee on your account.

Based on our fee schedule, this appears to be a **Monthly Maintenance Fee** which is charged when your account balance falls below $500.

**Good news!** There are several ways to waive this fee:
1. Maintain a $500 minimum daily balance
2. Set up direct deposit
3. Link a qualifying credit card

Would you like me to review your account to see if you qualify for a waiver?"""
    elif "password" in query.lower() or "login" in query.lower() or "access" in query.lower():
        response = """I can help you with account access.

For security, I'll need to verify your identity. Please confirm:
- Last 4 digits of your SSN, OR
- Date of birth AND zip code

Once verified, I can:
1. Send a password reset link to your registered email
2. Set up a temporary access code via SMS
3. Walk you through our self-service reset

Which option works best for you?"""
    elif "transfer" in query.lower() or "wire" in query.lower():
        response = """I can help with your transfer inquiry.

**Transfer Timelines:**
- ACH transfers: 1-3 business days
- Domestic wire: Same day if submitted before 5 PM EST
- International wire: 2-5 business days

**Pending transfers** can be cancelled before the end-of-day processing window.

Would you like me to check the status of a specific transfer?"""
    elif should_escalate:
        response = """I understand your frustration, and I want to make sure you get the help you deserve.

I'm going to connect you with a senior specialist who has more authority to resolve this for you. They'll be able to review your case thoroughly.

Please hold for just a moment while I brief them on your situation."""
    else:
        response = f"""Thank you for your question.

{context[:500] if context else "Let me look into this for you."}

Is there anything specific about this I can clarify?"""

    return response, cost, should_escalate


def display_agent_card(agent_name: str, status: str, cost: float = 0):
    """Display an agent card with status."""
    colors = {
        "active": "🟢",
        "waiting": "🟡",
        "done": "⚪",
    }
    rates = {
        "L1_Support": 18,
        "L2_Support": 25,
        "L3_Expert": 40,
    }

    icon = colors.get(status, "⚪")
    rate = rates.get(agent_name, 20)

    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin: 5px 0;">
        <strong>{icon} {agent_name}</strong><br>
        <small>Rate: ${rate}/hr | Cost: ${cost:.4f}</small>
    </div>
    """, unsafe_allow_html=True)


# Sidebar - System Status
with st.sidebar:
    st.header("🎧 System Status")

    st.subheader("Session Info")
    st.code(f"Session: {st.session_state.session_id}")
    st.code(f"Customer: {st.session_state.customer_id}")

    st.subheader("Costs")
    st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")

    st.subheader("Escalation Chain")
    if st.session_state.escalation_chain:
        st.write(" → ".join(st.session_state.escalation_chain))
    else:
        st.write("No escalations")

    st.divider()

    if st.button("🔄 Reset Session"):
        st.session_state.messages = []
        st.session_state.total_cost = 0.0
        st.session_state.escalation_chain = []
        st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.rerun()

    st.divider()

    st.subheader("Knowledge Base")
    try:
        from knowledge.document_loader import load_documents_from_directory
        data_dir = Path(__file__).parent / "data" / "grounding"
        docs = load_documents_from_directory(data_dir)
        categories = {}
        for d in docs:
            categories[d.category] = categories.get(d.category, 0) + 1

        for cat, count in categories.items():
            st.write(f"📁 {cat}: {count} docs")
    except Exception as e:
        st.write(f"⚠️ {e}")


# Main content
st.title("🎧 AI Customer Support Demo")
st.markdown("*Multi-Agent System with RAG, Persistent Memory, and Evaluation*")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🔍 Knowledge Explorer", "📊 Analytics", "🎤 Voice (Coming Soon)"])

with tab1:
    st.subheader("Customer Support Chat")

    # Agent pipeline visualization
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**L1 Support**")
        st.caption("Handles routine inquiries")
    with col2:
        st.markdown("**L2 Support**")
        st.caption("Technical issues")
    with col3:
        st.markdown("**L3 Expert**")
        st.caption("Complex cases")

    st.divider()

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                with st.expander("Details"):
                    st.json(message["metadata"])

    # Chat input
    if prompt := st.chat_input("How can we help you today?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # Step 1: Classify
                task_type, estimate = classify_query(prompt)

                # Step 2: Route
                specialists = get_specialists(task_type)

                # Step 3: Get context
                context = get_knowledge_context(prompt)

                # Step 4: Generate response
                current_agent = "L1_Support"
                response, cost, should_escalate = simulate_agent_response(prompt, current_agent, context)

                # Handle escalation
                if should_escalate:
                    st.session_state.escalation_chain.append(current_agent)
                    current_agent = "L2_Support"
                    st.session_state.escalation_chain.append(current_agent)
                    response, additional_cost, _ = simulate_agent_response(prompt, current_agent, context)
                    cost += additional_cost

                st.session_state.total_cost += cost

                # Display response
                st.markdown(response)

                # Show metadata
                metadata = {
                    "task_type": task_type,
                    "estimated_duration": f"{estimate}s",
                    "specialists": specialists,
                    "agent": current_agent,
                    "cost": f"${cost:.4f}",
                    "context_used": len(context) > 0,
                }

                with st.expander("🔍 Processing Details"):
                    st.json(metadata)

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "metadata": metadata,
        })


with tab2:
    st.subheader("Knowledge Base Explorer")

    search_query = st.text_input("Search knowledge base:", placeholder="e.g., monthly fee waiver")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("**Filter by Category:**")
        categories = ["All", "faq", "policy", "procedure", "fee_schedule", "product"]
        selected_category = st.radio("Category", categories)

    with col2:
        if search_query:
            try:
                from knowledge import KnowledgeRetriever
                data_dir = Path(__file__).parent / "data" / "grounding"
                retriever = KnowledgeRetriever(data_directory=data_dir)

                cat_filter = None if selected_category == "All" else [selected_category]
                results = retriever.retrieve(search_query, k=5, categories=cat_filter)

                if results:
                    st.success(f"Found {len(results)} relevant documents")
                    for i, r in enumerate(results):
                        with st.expander(f"📄 {r.document.title} (Score: {r.score:.2f})"):
                            st.markdown(f"**Category:** {r.document.category}")
                            st.markdown(f"**Source:** {r.document.source}")
                            st.markdown("---")
                            st.markdown(r.document.content)
                else:
                    st.warning("No results found")
            except Exception as e:
                st.error(f"Search error: {e}")
        else:
            st.info("Enter a search query to explore the knowledge base")


with tab3:
    st.subheader("Session Analytics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    with col3:
        st.metric("Escalations", len(st.session_state.escalation_chain))
    with col4:
        avg_cost = st.session_state.total_cost / max(len([m for m in st.session_state.messages if m["role"] == "assistant"]), 1)
        st.metric("Avg Cost/Response", f"${avg_cost:.4f}")

    st.divider()

    st.markdown("### Conversation Flow")

    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f"**{i+1}. Customer:** {msg['content'][:50]}...")
            else:
                metadata = msg.get("metadata", {})
                agent = metadata.get("agent", "Unknown")
                task_type = metadata.get("task_type", "Unknown")
                st.markdown(f"**{i+1}. {agent}:** [{task_type}] Response provided")
    else:
        st.info("Start a conversation to see analytics")


with tab4:
    st.subheader("🎤 Voice Interface (Preview)")

    st.warning("⚠️ Voice functionality requires additional setup (Ollama + Chatterbox TTS)")

    st.markdown("""
    ### How Voice Would Work:

    1. **Speech-to-Text (STT)**
       - User speaks → Whisper/Deepgram transcribes
       - Real-time streaming for low latency

    2. **Agent Processing**
       - Same multi-agent pipeline as chat
       - Context from knowledge base
       - Specialist routing

    3. **Text-to-Speech (TTS)**
       - Response → Chatterbox/ElevenLabs
       - Natural voice output

    ### Current Voice Support:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**TTS (Text-to-Speech)**")
        st.markdown("✅ Chatterbox integration available")
        st.code("python main.py --mode cli --voice", language="bash")

    with col2:
        st.markdown("**STT (Speech-to-Text)**")
        st.markdown("🚧 Not yet implemented")
        st.markdown("Would need: Whisper, Deepgram, or AssemblyAI")

    st.divider()

    st.markdown("### Test TTS Output")
    tts_text = st.text_area("Enter text to synthesize:", value="Hello, thank you for calling. How can I help you today?")

    if st.button("🔊 Synthesize (requires Chatterbox)", disabled=True):
        st.info("TTS synthesis would happen here")


# Footer
st.divider()
st.caption("Built with Streamlit, AutoGen AG2, LangChain, and FAISS")
