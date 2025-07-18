# === base_agents.py =========================================================
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
# from memory import log_interaction
from sim_agents import TaskAwareGroupChat, TaskAwareAgent
from task_classifier import classify

# ---------------------------------------------------------------------------
# 1.  SYSTEM MESSAGES
# ---------------------------------------------------------------------------
L1_SYS_MSG = (
   """
You are a Level 1 Customer Support Agent.

Your responsibilities include:
- Confer with customers by phone or message to provide product or service information
- Take or enter orders, cancel accounts, and update address or billing details
- Record inquiries, complaints, or comments, and actions taken
- Check that proper changes were made to resolve customer problems

Stay polite, efficient, and helpful. If a task is beyond your training or seems unresolved, escalate it to Level 2. Do not guess or fabricate information.
"""
)
L2_SYS_MSG = (
    """ You are a Level 2 Technical Support Agent.
    Your responsibilities include:
- Resolve billing complaints through refunds, adjustments, or service fixes
- Review insurance or account policy terms to determine appropriate coverage
- Con""tact customers with investigation results or next steps
- Use diagnostic reasoning and available tools to resolve escalated issues

You receive only unresolved queries from L1. If a problem requires deep system access, policy change, or expert investigation, escalate it to Level 3.
"""
)
L3_SYS_MSG = (
    """
You are a Level 3 Expert Support Engineer.

You specialize in complex and edge-case resolutions. Your responsibilities include:
- Examine disputed charges or technical failures by investigating full context
- Validate customer complaints through internal systems and data analysis
- Recommend improvements in product design, policy, or service workflow
- Coordinate with backend or engineering teams if needed

Only respond to cases that could not be resolved by Level 1 or 2 agents. Apply advanced reasoning, domain knowledge, and analytical skill to deliver a final resolution or root cause explanation.
"""

)

# ---------------------------------------------------------------------------
# 2.  LLM CONFIG
# ---------------------------------------------------------------------------
def get_ollama_config(model_name="mistral"):
    return {
        "config_list": [{
            "model": model_name,
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        }],
        "temperature": 0.3,
    }

# ---------------------------------------------------------------------------
# 3.  RAW AGENTS
# ---------------------------------------------------------------------------
l1_agent = AssistantAgent("L1_Support", system_message=L1_SYS_MSG,
                          llm_config=get_ollama_config("llama3"))
l2_agent = AssistantAgent("L2_Support", system_message=L2_SYS_MSG,
                          llm_config=get_ollama_config("gemma"))
l3_agent = AssistantAgent("L3_Expert",  system_message=L3_SYS_MSG,
                          llm_config=get_ollama_config("mistral"))

customer_proxy = UserProxyAgent(
    name="Customer",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)

# ---------------------------------------------------------------------------
# 4.  WRAPPED AGENTS (per‑task cost + timing)
# ---------------------------------------------------------------------------
L1_WRAPPED = TaskAwareAgent(l1_agent, role="L1", hourly_wage=18.0)
L2_WRAPPED = TaskAwareAgent(l2_agent, role="L2", hourly_wage=25.0)
L3_WRAPPED = TaskAwareAgent(l3_agent, role="L3", hourly_wage=40.0)

# ---------------------------------------------------------------------------
# 5.  ESCALATION & SUPPORT FLOW
# ---------------------------------------------------------------------------
_ESCALATE_TRIGGERS = [
    "escalate", "cannot resolve", "need help from l2",
    "pass this to", "not sure", "unresolved",
]

def _needs_escalation(txt: str) -> bool:
    low = txt.lower()
    return any(word in low for word in _ESCALATE_TRIGGERS)

def run_support_flow(user_query: str) -> str:
    """1‑on‑1 tiered support (L1→L2→L3)."""
    task_type, est_sec = classify(user_query)

    # --- L1 ---------------------------------------------------------------
    reply = L1_WRAPPED.handle(user_query, customer_proxy, task_type, est_sec)
    if not _needs_escalation(reply):
        return reply

    # --- L2 ---------------------------------------------------------------
    reply = L2_WRAPPED.handle(user_query, customer_proxy, task_type, int(est_sec * 1.25))
    if not _needs_escalation(reply):
        return reply

    # --- L3 ---------------------------------------------------------------
    return L3_WRAPPED.handle(user_query, customer_proxy, task_type, int(est_sec * 1.5))

# ---------------------------------------------------------------------------
# 6.  GROUP‑CHAT FLOW  (uses the same TaskAware cost accounting)
# ---------------------------------------------------------------------------
def run_groupchat_flow(user_query: str) -> str:
    task_type, est_sec = classify(user_query)

    gc = GroupChat(
        agents=[customer_proxy, l1_agent, l2_agent, l3_agent],
        messages=[],
        max_round=4,            # shave time — fewer rounds
    )
    manager = GroupChatManager(gc, llm_config=get_ollama_config("mistral"))
    wrapper = TaskAwareGroupChat(manager, role="GroupChat", hourly_wage=28.0)
    return wrapper.handle(user_query, customer_proxy, task_type, est_sec)
