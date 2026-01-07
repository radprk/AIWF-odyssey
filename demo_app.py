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


def display_agent_message(agent_name: str, message: str, msg_type: str = "thinking"):
    """Display a styled agent message in the conversation flow."""
    colors = {
        "L1_Support": "#4CAF50",      # Green
        "L2_Support": "#FF9800",       # Orange
        "L3_Expert": "#F44336",        # Red
        "Router": "#2196F3",           # Blue
        "Classifier": "#9C27B0",       # Purple
        "Knowledge": "#00BCD4",        # Cyan
    }

    icons = {
        "L1_Support": "🟢",
        "L2_Support": "🟡",
        "L3_Expert": "🔴",
        "Router": "🔀",
        "Classifier": "🏷️",
        "Knowledge": "📚",
    }

    color = colors.get(agent_name, "#607D8B")
    icon = icons.get(agent_name, "🤖")

    if msg_type == "handoff":
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {color}22, transparent);
                    border-left: 4px solid {color};
                    padding: 10px 15px;
                    margin: 10px 0;
                    border-radius: 0 8px 8px 0;">
            <strong>{icon} {agent_name}</strong> → <em style="color: #666;">{message}</em>
        </div>
        """, unsafe_allow_html=True)
    elif msg_type == "action":
        st.markdown(f"""
        <div style="background: {color}11;
                    border: 1px solid {color}44;
                    padding: 12px 15px;
                    margin: 8px 0;
                    border-radius: 8px;">
            <strong>{icon} {agent_name}:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:  # thinking
        st.markdown(f"""
        <div style="background: #f5f5f5;
                    padding: 8px 12px;
                    margin: 5px 0 5px 20px;
                    border-radius: 6px;
                    font-size: 0.9em;
                    color: #666;">
            💭 <em>{agent_name}: {message}</em>
        </div>
        """, unsafe_allow_html=True)

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
    except ImportError:
        # Fallback: read from raw files if langchain not installed
        return _fallback_knowledge_search(query)
    except Exception as e:
        return f"(Knowledge retrieval unavailable: {e})"


def _fallback_knowledge_search(query: str) -> str:
    """Simple keyword-based search when langchain is not available."""
    data_dir = Path(__file__).parent / "data" / "grounding"
    if not data_dir.exists():
        return ""

    query_lower = query.lower()
    keywords = query_lower.split()
    results = []

    # Search through all text/md/json files
    for ext in ["*.txt", "*.md", "*.json"]:
        for f in data_dir.rglob(ext):
            try:
                content = f.read_text()
                # Simple keyword matching
                matches = sum(1 for kw in keywords if kw in content.lower())
                if matches > 0:
                    results.append((matches, f.name, content[:300]))
            except Exception:
                continue

    # Sort by match count and return top results
    results.sort(reverse=True, key=lambda x: x[0])

    if results:
        context_parts = []
        for _, name, preview in results[:3]:
            context_parts.append(f"[From {name}]: {preview}...")
        return "\n\n".join(context_parts)

    return ""


def process_with_agents(query: str, show_flow_callback) -> dict:
    """
    Process query through the multi-agent system with visible communication.
    Returns dict with response, costs, agents involved, and communication log.
    """
    communication_log = []
    total_cost = 0.0
    agents_involved = []

    # Agent hourly rates
    rates = {
        "L1_Support": 18.0,
        "L2_Support": 25.0,
        "L3_Expert": 40.0,
        "Billing_Specialist": 22.0,
        "Account_Specialist": 22.0,
    }

    # === STEP 1: Classifier Agent ===
    show_flow_callback("Classifier", "Analyzing incoming query...", "action")
    time.sleep(0.3)

    task_type, estimate = classify_query(query)
    communication_log.append({
        "from": "Classifier",
        "to": "Router",
        "message": f"Query classified as '{task_type}' (estimated {estimate}s to resolve)"
    })
    show_flow_callback("Classifier", f"Classified as: {task_type}", "thinking")
    time.sleep(0.2)

    # === STEP 2: Router Agent ===
    show_flow_callback("Router", "Determining specialist routing...", "action")
    time.sleep(0.3)

    specialists = get_specialists(task_type)
    communication_log.append({
        "from": "Router",
        "to": "L1_Support",
        "message": f"Routing to specialists: {', '.join(specialists)}"
    })
    show_flow_callback("Router", f"Specialists selected: {', '.join(specialists)}", "thinking")
    time.sleep(0.2)

    # === STEP 3: Knowledge Retrieval ===
    show_flow_callback("Knowledge", "Searching knowledge base for relevant context...", "action")
    time.sleep(0.3)

    context = get_knowledge_context(query)
    context_preview = context[:100] + "..." if len(context) > 100 else context
    communication_log.append({
        "from": "Knowledge",
        "to": "L1_Support",
        "message": f"Found context: {context_preview}"
    })
    show_flow_callback("Knowledge", f"Retrieved {len(context)} chars of context", "thinking")
    time.sleep(0.2)

    # === STEP 4: L1 Support Agent ===
    show_flow_callback("L1_Support", "Receiving query and context...", "handoff")
    time.sleep(0.3)

    # Check for escalation triggers
    escalation_keywords = ["supervisor", "manager", "escalate", "unacceptable", "lawyer", "sue", "third time", "again and again"]
    high_escalation = ["lawyer", "sue", "legal", "court"]
    should_escalate = any(kw in query.lower() for kw in escalation_keywords)
    needs_l3 = any(kw in query.lower() for kw in high_escalation)

    agents_involved.append("L1_Support")
    l1_time = 3.0
    l1_cost = (l1_time / 3600) * rates["L1_Support"]
    total_cost += l1_cost

    if should_escalate:
        # L1 decides to escalate
        show_flow_callback("L1_Support", "Analyzing sentiment... detecting frustration indicators", "thinking")
        time.sleep(0.3)

        communication_log.append({
            "from": "L1_Support",
            "to": "L2_Support",
            "message": "ESCALATION REQUEST: Customer expressing frustration, requesting supervisor. Transferring with full context."
        })
        show_flow_callback("L1_Support", "Initiating escalation to L2 Support...", "handoff")
        time.sleep(0.3)

        # === STEP 5: L2 Support Agent (Escalation) ===
        show_flow_callback("L2_Support", "Accepting escalation from L1...", "action")
        time.sleep(0.3)

        agents_involved.append("L2_Support")
        l2_time = 5.0
        l2_cost = (l2_time / 3600) * rates["L2_Support"]
        total_cost += l2_cost

        communication_log.append({
            "from": "L2_Support",
            "to": "L1_Support",
            "message": "ACK: Escalation accepted. I have authority to offer compensation/resolution."
        })
        show_flow_callback("L2_Support", "Reviewing escalation context and customer history...", "thinking")
        time.sleep(0.3)

        if needs_l3:
            # Further escalate to L3
            communication_log.append({
                "from": "L2_Support",
                "to": "L3_Expert",
                "message": "CRITICAL ESCALATION: Legal concerns mentioned. Requires expert handling."
            })
            show_flow_callback("L2_Support", "Legal concerns detected - escalating to L3 Expert...", "handoff")
            time.sleep(0.3)

            show_flow_callback("L3_Expert", "Accepting critical escalation...", "action")
            agents_involved.append("L3_Expert")
            l3_time = 8.0
            l3_cost = (l3_time / 3600) * rates["L3_Expert"]
            total_cost += l3_cost

            response = """I'm a senior resolution specialist, and I've been briefed on your situation.

I want to assure you that we take your concerns very seriously. I have full authority to resolve this matter today.

After reviewing your account, I can see the charges you're referring to. Let me offer the following resolution:

1. **Immediate refund** of all disputed charges
2. **Account credit** of $25 for the inconvenience
3. **Direct line** to our resolution team for any future concerns

I'll also personally ensure this issue is documented to prevent recurrence. Would this resolution work for you?"""

        else:
            response = """I'm a senior support specialist, and I've taken over your case from our front-line team.

I understand you're frustrated, and I apologize for any inconvenience. I have more authority to resolve issues and can offer solutions that my colleague couldn't.

Looking at your account and the situation, here's what I can do:

1. **Investigate** the root cause of this issue
2. **Provide compensation** if our error is confirmed
3. **Escalate further** if needed to our management team

Let me look into this right now. Can you give me a moment to review your full account history?"""

    else:
        # Standard L1 handling (no escalation)
        show_flow_callback("L1_Support", "Processing query with available context...", "thinking")
        time.sleep(0.3)

        # Generate response based on query type
        if "fee" in query.lower() and ("15" in query or "charge" in query.lower()):
            response = """I understand you're concerned about the fee on your account.

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

        else:
            response = f"""Thank you for your question.

{context[:500] if context else "Let me look into this for you."}

Is there anything specific about this I can clarify?"""

        communication_log.append({
            "from": "L1_Support",
            "to": "Customer",
            "message": "Response generated successfully"
        })

    # Final agent delivers response
    final_agent = agents_involved[-1]
    show_flow_callback(final_agent, "Delivering response to customer...", "action")

    return {
        "response": response,
        "total_cost": total_cost,
        "agents_involved": agents_involved,
        "communication_log": communication_log,
        "task_type": task_type,
        "specialists": specialists,
        "context_length": len(context),
        "escalated": len(agents_involved) > 1,
    }


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

        # Process query with visible agent flow
        with st.chat_message("assistant"):
            # Create a container for agent communication flow
            flow_container = st.container()

            with flow_container:
                st.markdown("### 🔄 Agent Communication Flow")
                st.markdown("---")

                # Placeholder for agent messages
                agent_flow_area = st.empty()
                flow_messages = []

                def show_flow(agent_name, message, msg_type):
                    """Callback to display agent communication in real-time."""
                    flow_messages.append((agent_name, message, msg_type))
                    with agent_flow_area.container():
                        for a, m, t in flow_messages:
                            display_agent_message(a, m, t)

                # Process through multi-agent system
                result = process_with_agents(prompt, show_flow)

                st.markdown("---")

                # Show agent-to-agent message log
                if result["communication_log"]:
                    with st.expander("📨 Agent-to-Agent Messages", expanded=True):
                        for msg in result["communication_log"]:
                            st.markdown(f"""
                            <div style="background: #f0f7ff; padding: 8px 12px; margin: 5px 0; border-radius: 6px; border-left: 3px solid #2196F3;">
                                <strong>{msg['from']}</strong> → <strong>{msg['to']}</strong><br>
                                <span style="color: #555;">{msg['message']}</span>
                            </div>
                            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 💬 Response")

            # Update session state
            st.session_state.total_cost += result["total_cost"]
            if result["escalated"]:
                st.session_state.escalation_chain.extend(result["agents_involved"])

            # Display response
            st.markdown(result["response"])

            # Show metadata
            metadata = {
                "task_type": result["task_type"],
                "specialists": result["specialists"],
                "agents_involved": result["agents_involved"],
                "escalated": result["escalated"],
                "cost": f"${result['total_cost']:.4f}",
                "context_length": result["context_length"],
            }

            with st.expander("🔍 Processing Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Cost", f"${result['total_cost']:.4f}")
                    st.metric("Agents Used", len(result["agents_involved"]))
                with col2:
                    st.metric("Escalated", "Yes" if result["escalated"] else "No")
                    st.metric("Context Chars", result["context_length"])

                st.markdown("**Agent Chain:**")
                st.markdown(" → ".join(result["agents_involved"]))

                st.json(metadata)

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "metadata": metadata,
            "communication_log": result["communication_log"],
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
