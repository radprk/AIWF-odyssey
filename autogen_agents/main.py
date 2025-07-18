# === main.py ===============================================================
from simulation_engine import simulate_query_handling
from base_agents import run_groupchat_flow   # <- already has wrapped agents

def _batch_demo() -> None:
    print("\n🚀  Running autonomous GroupChat batch …\n")
    questions = [
        "Why was I charged a $15 fee?",
        "Can I cancel a pending transfer?",
        "What is a margin call?",
        "How do I dispute a charge?",
    ]
    for i, q in enumerate(questions, 1):
        print(f"\n== Query {i} ==\n")
        run_groupchat_flow(q)
    print("\n✅  GroupChat simulation complete!\n")

def _interactive_cli() -> None:
    print("\n🗣️  Enter support queries (type 'exit' to quit)\n")
    qid = 1
    while True:
        text = input("🧑‍💻 You: ").strip()
        if text.lower() in {"exit", "quit"}:
            print("👋  Bye!")
            break
        simulate_query_handling(qid, text)   # uses run_support_flow internally
        qid += 1

if __name__ == "__main__":
    print("Customer Support Simulator")
    mode = input("Choose mode [1] CLI | [2] Batch demo: ").strip()
    _batch_demo() if mode == "2" else _interactive_cli()
