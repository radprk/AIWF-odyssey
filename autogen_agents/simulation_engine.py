# === simulation_engine.py ===
import random
import time
from datetime import datetime
from base_agents import run_support_flow, run_routed_support_flow
from faq_loader import load_faq_database

# Load FAQ grounding once globally
faq_db = load_faq_database("../data/grounding/faq.txt")

# Agent configuration
AGENT_TIERS = {
    "L1": {"count": 3, "avg_time": 6},
    "L2": {"count": 2, "avg_time": 10},
    "L3": {"count": 1, "avg_time": 15},
}

# Agent pools (number available per tier)
agent_pool = {tier: AGENT_TIERS[tier]["count"] for tier in AGENT_TIERS}

# Global logs
call_log = []

def simulate_query_handling(query_id, query_text, complexity="medium", session_id: str | None = None,
                            use_router: bool = False, use_llm_router: bool = False):
    start_time = datetime.now()
    print(f"\n Handling Query {query_id}: {query_text} [{complexity}]")

    # === Search FAQ DB for grounding context ===
    # retrieved_docs = faq_db.similarity_search(query_text, k=3)
    # grounding_context = "\n".join([doc.page_content for doc in retrieved_docs])
    # augmented_query = (
    #     f"You are grounded in the following FAQ context:\n\n{grounding_context}\n\n"
    #     f"Now answer the user's question:\n\n{query_text}"
    # )

    # Simulate L1 attempt
    if agent_pool["L1"] > 0:
        agent_pool["L1"] -= 1
        print(" Routed to L1 agent...")
        time.sleep(random.gauss(AGENT_TIERS["L1"]["avg_time"], 2))
        if use_router:
            response = run_routed_support_flow(
                query_text,
                session_id=session_id,
                use_llm_router=use_llm_router,
            )
        else:
            response = run_support_flow(query_text, session_id=session_id)

        agent_pool["L1"] += 1
    else:
        print(" L1 busy... waiting")
        time.sleep(2)
        return simulate_query_handling(query_id, query_text, complexity)

    # Check for escalation
    if "escalate" in response.lower() or "unresolved" in response.lower():
        if agent_pool["L2"] > 0:
            agent_pool["L2"] -= 1
            print(" Escalated to L2 agent...")
            time.sleep(random.gauss(AGENT_TIERS["L2"]["avg_time"], 2))
            response = run_support_flow(query_text, session_id=session_id)
            agent_pool["L2"] += 1

            if "escalate" in response.lower():
                if agent_pool["L3"] > 0:
                    agent_pool["L3"] -= 1
                    print("🚨 Escalated to L3 expert...")
                    time.sleep(random.gauss(AGENT_TIERS["L3"]["avg_time"], 2))
                    response = run_support_flow(query_text, session_id=session_id)
                    agent_pool["L3"] += 1
                else:
                    response += "\n[L3 unavailable, please try again later.]"
        else:
            response += "\n[L2 unavailable, please try again later.]"

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    call_log.append({
        "id": query_id,
        "query": query_text,
        "session_id": session_id,
        "start": start_time,
        "end": end_time,
        "duration": elapsed,
        "final_response": response,
    })
    print(f" Resolved in {elapsed:.2f}s — Response Summary:\n{response[:300]}...\n")
    return response
